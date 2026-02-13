const CACHE_NAME = "DailyCompanion-v1";
const ASSETS_TO_CACHE = [
  "./",
  "./index.html",
  "./favicon.ico"
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
  // Firebase/Firestore Anfragen immer ignorieren (Live-Daten)
  if (event.request.url.includes("firestore") || event.request.url.includes("googleapis")) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Wenn Netzwerk-Antwort OK, Kopie in den Cache legen (optional)
        // und die Antwort zurückgeben
        return caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, response.clone());
          return response;
        });
      })
      .catch(() => {
        // Wenn das Netzwerk fehlschlägt, nimm die Version aus dem Cache
        return caches.match(event.request);
      })
  );
});
