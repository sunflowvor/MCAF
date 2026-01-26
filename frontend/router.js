export function getRoute() {
  return window.location.hash.replace("#", "") || "/";
}

export function nav(path) {
  window.location.hash = path;
}