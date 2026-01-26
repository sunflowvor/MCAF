from utils.inference.memory_manager import MemoryManager
from utils.tracking import XMem
from utils.tracking_utils import aggregate

import torch.nn.functional as F
#from util.tensor_util import pad_divide_by, unpad

# STM
def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if len(img.shape) == 4:
        if pad[2]+pad[3] > 0:
            img = img[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,:,pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2]+pad[3] > 0:
            img = img[:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img

class InferenceCore:
    def __init__(self, network:XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        multi_scale_features = (f16, f8, f4)

        # segment the current frame is needed
        if need_segment:
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
            hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
            if is_normal_update:
                self.memory.set_hidden(hidden)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)

            # also create new hidden states
            self.memory.create_hidden_state(len(self.all_labels), key)

        # save as memory if needed
        if is_mem_frame:
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)
            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                    selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)

import numpy as np
import torch

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)
        
    return Ms
class MaskMapper:
    """
    This class is used to convert a indexed-mask to a one-hot representation.
    It also takes care of remapping non-continuous indices
    It has two modes:
        1. Default. Only masks with new indices are supposed to go into the remapper.
        This is also the case for YouTubeVOS.
        i.e., regions with index 0 are not "background", but "don't care".

        2. Exhaustive. Regions with index 0 are considered "background".
        Every single pixel is considered to be "labeled".
    """
    def __init__(self):
        self.labels = []
        self.remappings = {}

        # if coherent, no mapping is required
        self.coherent = True

    def convert_mask(self, mask, exhaustive=False):
        # mask is in index representation, H*W numpy array
        labels = np.unique(mask).astype(np.uint8)
        labels = labels[labels!=0].tolist()

        new_labels = list(set(labels) - set(self.labels))
        if not exhaustive:
            assert len(new_labels) == len(labels), 'Old labels found in non-exhaustive mode'

        # add new remappings
        for i, l in enumerate(new_labels):
            self.remappings[l] = i+len(self.labels)+1
            if self.coherent and i+len(self.labels)+1 != l:
                self.coherent = False

        if exhaustive:
            new_mapped_labels = range(1, len(self.labels)+len(new_labels)+1)
        else:
            if self.coherent:
                new_mapped_labels = new_labels
            else:
                new_mapped_labels = range(len(self.labels)+1, len(self.labels)+len(new_labels)+1)

        self.labels.extend(new_labels)
        mask = torch.from_numpy(all_to_onehot(mask, self.labels)).float()

        # mask num_objects*H*W
        return mask, new_mapped_labels


    def remap_index_mask(self, mask):
        # mask is in index representation, H*W numpy array
        if self.coherent:
            return mask

        new_mask = np.zeros_like(mask)
        for l, i in self.remappings.items():
            new_mask[mask==i] = l
        return new_mask