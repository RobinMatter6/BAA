# BAA Sprint Abgleichsprotokolle


Im Folgenden werden die Sprint-Abgleiche bezüglich der Bachelorarbeit zwischen Robin Matter und Ramón Christen festgehalten:

<details>
<br>
  <summary><strong>Protokoll des Kickoff Meeting</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 15. September 2024  
  **Themen:** Projektplanung, Aufgabenaufstellung, Literaturrecherche, Datenzugang und Dokumentation


---

### 1. Ziel des Meetings
Festlegung des Projektaufbaus, Aufgabenaufstellung und Zeitabschätzung, sowie Klärung offener Fragen.

### 2. Besprochene Themen und Beschlüsse

#### Projekt-Management
Es sollte eine erste Projektplanung mit folgenden Elementen bis zum nächsten Abgleich gemacht werden:
- Aufgabenstellung fix definieren
- Meilensteine definieren
- Zeitlicher Plan mit Meilensteinen bis zum Projektende erstellen
- Risikoanalyse durchführen

#### Literaturrecherche
- **Recherchequellen**: Für die Literaturrecherche sollte vor allem bei folgenden Quellen gesucht werden:
  - Google Scholar
  - IEEE Xplore
  - Gate Research
  - Arxiv OpenLibrary (akzeptiert, jedoch andere vorziehen, da Arxiv nicht peer-reviewed ist)
  - Fachbücher

#### Wetterdatenzugang
- **MeteoSchweiz Zugang**: Ramón Christen stellte einen Link für den Zugang zu Wetterdaten von MeteoSchweiz bereit.
<br><br>
</details>


<details>
<br>
  <summary><strong>Protokoll des 1. Oktober 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 1. Oktober 2024  
  **Themen:** Datenzugang, Vorverarbeitung und Analyse
  

---

### 1. Ziel des Meetings
Etablierung des Datenzugangs, erste Schritte der Vorverarbeitung und Diskussion der geplanten Modellansätze.

### 2. Besprochene Themen und Beschlüsse

#### Forschungsfragen
- **Erster Draft:** Es soll ein erster Draft der Forschungsfragen erarbeitet werden. Diese Fragen sollen die Bachelorarbeit stützen und idealerweise aufeinander aufbauen.

#### Literaturrecherche
- **Suchbegriff:** Ramón Christen empfahl für die Recherche nach Temporal Saliency Detection zu suchen, da dies ein verbreiter Überbegriff des Themas darstellt.
- **Towards DataScience:** Es wurde empfohlen, sich bei technischen Fragen auf Towards DataScience umzusehen. Für die Dokumentation sollten diese Quellen jedoch nicht verwendet werden.

#### Datenaufbereitung und Analyse
- **Meteo Datenzugang:** Es wurden als nächste Schritte definiert abzuwarten bis Meteo Datenzugriff erhalten wird und dann mit der Datenverabeitung zu beginnen.
- **Libraries:** Es wurde empfohlen mit Python und den Libraries Pandas, Numpy und MatPlotLib zu arbeiten.
<br><br>
</details>



<details>
<br>
  <summary><strong>Protokoll des 15. Oktober 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 15. Oktober 2024  
  **Themen:** Datenzugang, Daten Vorverarbeitung und Analyse



---



### 1. Ziel des Meetings
Abklärung des Fortschritts in der Datenaufbereitung und Analyse.

### 2. Besprochene Themen und Beschlüsse

#### Datenaufbereitung
- **Speicherform:** Die Ausgabe der erstellten Datenaufbereitungsklasse ist ein Python Dictionary mit einem Pandas Dataframe für jeden Tourismushot spot. Ramón Christen vermerkte, dass diese Form einfach in einem JSON-File gespeichert werden kann.
- **Outlier, Standardisierung und Skewness:** Es wurde entschieden, dass die Daten weiter bereinigt und aufbereitet werden müssen, um für das Training des Machine Learning Modells geeignet zu sein.

#### Datenanalyse
- **Datenanalysesplit:** Bis zum Meeting waren Auswertungen aufgeteilt nach jedem Wochentag möglich. Da die Datenmenge dadurch gering wurde und ähnliche Muster innerhalb der Arbeitswoche und des Wochenendes sichtbar sind, wurde entschieden, zusätzlich die Daten basierend auf Arbeitswoche und Wochenende zu analysieren. So sollten einige der Anomalien in der Datenanalyse beseitigt werden.
- **Anzahl der Kategorien reduzieren:** Bei einer Art der ausgearbeiteten Plots werden die jeweilige exogene Variable in verschiedene Kategorien aufgeteilt und in Zusammenhang mit dem Besucheraufkommen gestellt. Es lassen sich Muster erkennen, doch es bestehen Anomalien, die als nächster Schritt durch die Reduzierung der Anzahl der Kategorien vermindert werden sollten. Regen und Sonne sollten zum Beispiel nur noch in zwei Kategorien untersucht werden: vorhanden, ja oder nein.
#### TFT-Training
- **Saisonale Effekte:** Es wurde besprochen, dass saisonale Effekte wie die Fasnacht nicht berücksichtigt werden können, da nur Daten von ungefähr 1.5 Jahren vorhanden sind.
- **GPU-Hub:** Es wurde besprochen, dass der GPU-Hub der HSLU für das Training des Modells verwendet werden könnte.
#### Zwischenpräsentation
- **Termin:** Es wurde mögliche Termine für die Zwischenpräsentation zusammengetragen, um anschliessend Zeitvorschläge dem Experten zukommen zulassen.
<br><br>
</details>

<details>
<br>
  <summary><strong>Protokoll des 29. Oktober 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 29. Oktober 2024  
  **Themen:** Datenanalyse und TFT-Modellentwicklung


---
### 1. Ziel des Meetings
Klärung des aktuellen Standes und Definition der nächsten Schritte für die Datenanalyse und Modellentwicklung.

### 2. Besprochene Themen und Beschlüsse

#### TFT-Implementierung
- **PyTorch Forecasting:** Es wurde die erste Implementierung des TFT unter der Verwendung von PyTorch Forecasting gezeigt.
- **Google Research GitHub:** Es wurde entschieden, den Aufbau des TFT mittels PyTorch Forecasting nicht weiter fortzusetzen und stattdessen den TFT direkt vom [Google Research GitHub Repository](https://github.com/google-research/google-research/tree/master/tft) zu implementieren.
- **Attention:** Es sollte geprüft werden, wie die Wichtigkeit einzelner Ereignisse in den exogenen Variablen von dem trainierten TFT erhalten werden kann.

#### Datenaufbereitung
- **Probleme:** Robin äusserte Probleme beim Entfernen von Outliers und Schwierigkeiten beim Entfernen von Skewness in den Daten.  
- **Skewness:** Zur Behebung der Schiefe in den Daten einzelner exogenen Variablen wurde dazu geraten den Logarithmus zu verstärken.
- **Outlier:** Es wurde beschlossen nach einem IQR Prinzip mit einem 30 Minutenfenster für jeden Wochentag die Outlier zu entfernen.
- **Zeitumstellung:** Sommer-/Winterzeit sollte in der Datenvorverarbeitung berücksichtigt werden.

#### Trainingsdaten verbesserung
-**Statische Code Analyse:** Eine Gauss-Verteilung und Histogramm-Kategorisierung sollen zur Verbesserung der Trainingsdaten verwendet werden.

#### Zwischenpräsentation
- **Slides:** Erster Aufbau der PowerPoint Slides und geplanter Aufbau wurde gezeigt.
- **Termin:** Der Zwischenpräsentations-Termin wurde festgelegt.
<br><br>

> **Hinweis:** Die To-Dos sind als Issues in diesem Repository erfasst.

<br><br>
</details>

<details>
<br>
  <summary><strong>Protokoll des 12. November 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 12. November 2024  
  **Themen:** TFT-Modellentwicklung, Datenanalyse, Zwischenpräsentation und Dokumentation
  
---

### 1. Ziel des Meetings
Überprüfung des Fortschritts bezüglich der TFT-Inbetriebnahme, Outlier-/Skew-Entfernung und Zwischenpräsentationsvorbereitung.

### 2. Besprochene Themen und Beschlüsse

#### TFT-Modellentwicklung
- **DataLoader für eigene Daten**: Ein spezieller DataLoader für die eigenen Daten soll als nächstes entwickelt werden.
- **Erste Visualisierungen**: Erste Plots der Atention Wheights sollen gemacht werden.
- **Training**: Es wurde beschlossen ein Training ausschliesslich mit den *visitor*-Daten und ein Training mit allen Exogenen-Daten (ausser Schnee und Percipation(min) aufgrund von zu starkem skew) durchgeführt.



#### Datenvorverarbeitung und Analyse-Dokumentation
- **Dokumentation ohne Code**: Der Betreuer Ramón Christen forderte, dass in der Dokumentation kein Code verwendet wird. Falls notwendig, soll stattdessen Pseudocode verwendet werden. In diesem Zusammenhang sollte für die Datenaufbereitung und die Datenanalyse dokumentiert werden, was gemacht wurde. Dabei sollte jedoch nicht zu weit ins Detail gegangen werden.
- **Outlierentfernung nur für Visitor-Daten**: Es wurde entschieden die Outlier ausschliesslich bei den *visitor*-Daten zu entfernen.

#### Schlussabgabe
- **Code und Daten Bereitstellung**: Ramón Christen präferiert eine Code Abgabe basierend auf GitHub. Alternativ müsste auf eine ZIP-Datei ausgewichen werden. Aufgrund des Datenschutzes wurde festgelegt, dass die Daten nicht mitgeliefert werden müssen.

#### Zwischenpräsentation
- **Daniel Wechsler**: Es wurde beschlossen, Daniel Wechsler, der zuständig für die Bachelorarbeit bei der Partnerfirma "Arcade" ist, im CC für die Zwischenpräsentation einzuladen.
- **Präsentations-Bilder**: Bilder, die nicht ausschliesslich dekorativen Charakter haben, sollen mit "source <Quelle>" deklariert werden. Als Quellenangabe reicht dabei der Name der Quelle. Es wurde das Beispiel "source Wikipedia" genannt.
<br><br>

> **Hinweis:** Die To-Dos sind als Issues in diesem Repository erfasst.


<br><br>
</details>
