(async () => {
    console.log("Starte Datenexport...");
    
    try {
        // Zugriff auf die Firestore-Instanz, die bereits in deiner App definiert ist
        const snapshot = await db.collection("dailyEntries").get();
        
        if (snapshot.empty) {
            console.log("Keine Daten in 'dailyEntries' gefunden.");
            return;
        }

        const data = [];
        snapshot.forEach(doc => {
            const docData = doc.data();
            
            // Konvertierung von Firestore-Timestamps in lesbare ISO-Daten
            // (Damit das JSON nicht aus kryptischen Sekunden/Nanosekunden besteht)
            for (let key in docData) {
                if (docData[key] && typeof docData[key].toDate === 'function') {
                    docData[key] = docData[key].toDate().toISOString();
                }
            }
            
            data.push({
                id: doc.id,
                ...docData
            });
        });

        // Erstellung der JSON-Datei
        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        
        // Automatischer Download-Link
        const link = document.createElement("a");
        link.href = url;
        link.download = `daily_entries_export_${new Date().toISOString().slice(0,10)}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`Erfolg! ${data.length} Einträge wurden exportiert.`);
    } catch (error) {
        console.error("Fehler beim Exportieren der Daten:", error);
    }
})();
