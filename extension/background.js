// Chrome runs this as a service worker (loads via manifest's
// background.service_worker) and needs the polyfill pulled in manually;
// Firefox runs it as a plain background script (background.scripts),
// where the polyfill is already loaded first via the scripts array, and
// `importScripts` doesn't exist in that context at all.
if (typeof importScripts === "function") {
  importScripts("vendor/browser-polyfill.min.js");
}

browser.runtime.onInstalled.addListener(() => {
  console.log("Fact Check ML installed");
});
