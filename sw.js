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
  self.skipWaiting(); // Sofort aktivieren
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[Service Worker] Caching Assets");
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// 2. Aktivieren: Alte Caches löschen
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
  return self.clients.claim(); // Kontrolle über alle Tabs übernehmen
});

// 3. Fetch: Intelligentes Caching (Kein träges Network-First für die App-Shell)
self.addEventListener("fetch", (event) => {
  const url = event.request.url;

  // Firebase/Firestore komplett ignorieren (Live-Daten)
  if (url.includes("firestore") || url.includes("googleapis.com/google.firestore")) {
    return;
  }

  // Nur GET-Anfragen verarbeiten
  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      // 1. Wenn es im Cache ist, gib es SOFORT zurück (Blitzschneller App-Start)
      if (cachedResponse) {
        
        // Strategie für eigene Dateien (index.html etc.): Stale-While-Revalidate
        // Holt die Datei aus dem Cache, prüft aber im Hintergrund lautlos, ob es ein Update gibt.
        const isCdn = url.includes("cdn.jsdelivr") || url.includes("unpkg.com") || url.includes("npmcdn.com");
        if (!isCnt) {
          fetch(event.request).then((networkResponse) => {
            if (networkResponse.status === 200) {
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, networkResponse));
            }
          }).catch(() => /* Offline oder Netzwerkfehler im Hintergrund ignorieren */ {});
        }
        
        return cachedResponse;
      }

      // 2. Wenn nicht im Cache, normal aus dem Netzwerk laden und für das nächste Mal cachen
      return fetch(event.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200) {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });

        return networkResponse;
      }).catch(() => {
        // Optional: Hier könntest du eine Offline-Fallback-Seite ausliefern
      });
    })
  );
});
