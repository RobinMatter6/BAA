# BAA Sprint Vergleichsprotokolle

**Teilnehmer:** Robin Matter, Ramon Christen  
**Datum:** 29. Oktober 2024  
**Thema:** Besprechung des Fortschritts der Bachelorarbeit (Datenanalyse und TFT-Modellentwicklung)

---

## 1. Ziel des Meetings
Klärung des aktuellen Standes und Definition der nächsten Schritte für die Datenanalyse und Modellentwicklung.

## 2. Besprochene Themen und Beschlüsse

### TFT
Die TFT-Implementierung mit PyTorch Forecasting wurde gezeigt, und es wurde beschlossen, den TFT auf Basis des Google Research GitHub Repositorys aufzubauen:  
[https://github.com/google-research/google-research/tree/master/tft](https://github.com/google-research/google-research/tree/master/tft).  
Das Ziel ist es, zu prüfen, ob das Modell die Distanzinformationen bezüglich der Wichtigkeit von Ereignissen in den exogenen Variablen liefert. Falls dies nicht der Fall ist, sollten Alternativen in Betracht gezogen werden.

### Ausreisser und Schiefe
Robin äusserte Probleme beim Entfernen von Ausreissern und Schwierigkeiten beim Entfernen von Skewness in den Daten.  
Als Tipp wurde gegeben, einen stärkeren Logarithmus auf die Spalten anzuwenden, die noch zu stark von Skewness betroffen sind. Für die korrekte Entfernung der Ausreisser wurde beschlossen, diese für alle Wochentage mit einem Halbstundenfenster nach dem IQR-Prinzip zu entfernen. Darüber hinaus sollte mit unterschiedlich starken IQR-Outlier-Erkennungsverfahren gearbeitet werden.

### Literaturrecherche
Es sollte eine Recherche nach weiteren Papers zum Forschungsgebiet der BAA durchgeführt werden.

### Zeitumstellung
Sommer-/Winterzeit sollte in der Datenvorverarbeitung berücksichtigt werden.

### Statistische Analyse
Gauss-Verteilung und Histogramm-Kategorisierung sollen zur Verbesserung der Trainingsdaten verwendet werden.

> **Hinweis:** Die To-Dos sind als Issues in diesem Repository erfasst.

## 4. Nächstes Treffen
**Datum:** 12. November 2024
