const CACHE_NAME = "DailyCompanion-v2";
const ASSETS_TO_CACHE = [
  "./",
  "./index.html",
  "./favicon.ico",
  "https://cdn.jsdelivr.net/npm/chart.js",
  "https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns",
  "https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.2.1",
  "https://cdn.jsdelivr.net/npm/flatpickr",
  "https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css",
  "https://npmcdn.com/flatpickr/dist/l10n/de.js",
  "https://unpkg.com/simple-statistics@7.8.3/dist/simple-statistics.min.js"
];

// 1. Installieren: Dateien in den Cache laden
self.addEventListener("install", (event) => {
  // forceer das sofortige Übernehmen des neuen Workers
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[Service Worker] Caching neue Version:", CACHE_NAME);
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// 2. Aktivieren: Alte Caches (v1, etc.) löschen
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            console.log("[Service Worker] Lösche alten Cache:", key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  // Übernimmt sofort die Kontrolle über alle offenen Tabs
  return self.clients.claim();
});

// 3. Fetch: Netzwerk-Priorität (Network First, Falling Back to Cache)
self.addEventListener("fetch", (event) => {
  if (event.request.url.includes("firestore") || event.request.url.includes("googleapis.com/google.firestore")) {
    return; // Firebase live lassen
  }

  // Für Google Fonts und CDN-Assets: Cache First
  const isCdnAsset = event.request.url.includes("cdn.jsdelivr") || 
                     event.request.url.includes("unpkg.com") ||
                     event.request.url.includes("npmcdn.com") ||
                     event.request.url.includes("fonts.googleapis") ||
                     event.request.url.includes("fonts.gstatic");

  if (isCdnAsset) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        return cached || fetch(event.request).then(response => {
          return caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, response.clone());
            return response;
          });
        });
      })
    );
    return;
  }

  // Für eigene Dateien: Network First (wie bisher)
  event.respondWith(
    fetch(event.request)
      .then(response => {
        return caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, response.clone());
          return response;
        });
      })
      .catch(() => caches.match(event.request))
  );
});
