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


<details>
<br>
  <summary><strong>Protokoll des 26. November 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 26. November 2024  
  **Themen:** TFT-Paraemeter optimierung, TFT-Attention Auslesen, Datenaufbereitung, Dokumentaion
  
---

### 1. Ziel des Meetings
Besprechung der Zwischenpräsentation wie auch der erzielten Resultate, bezüglich des TFT-Trainings und des Attention-Auslesens und -Darstellens.


### 2. Besprochene Themen und Beschlüsse


#### TFT-Parameter optimieren
- **MAE-Test-Skript:** Es wurde entschieden, dass als nächster Schritt ein Testskript geschrieben werden sollte, das mittels MAE (Mean Absolute Error) die mit verschiedenen Parametern trainierten Modelle testet.
- **Erste Parameter Wahl:** Bei der Parameterwahl wurde bereits entschieden, dass das Modell die Daten der letzten Woche und des kommenden Tages berücksichtigen sollte. Wenn nötig, kann von dort verkleinert werden.


#### Hyperparameter tuning schlechter als nach Training:
- **Problem:** Robin teilte mit, dass bei den ersten Versuchen das Hyperparametertuning mit höherem Loss startete, als das Trainingsskript am Ende hatte. So war der Loss nach dem Hyperparametertuning höher als nach dem Training.
- **Vorschlag von Robin:** Robin schlug als nächsten Schritt vor, genauer zu untersuchen und sicherzustellen, dass das korrekte Modell für das Hyperparametertuning geladen wird.
- **Vermerk von Ramón Christen:** Darüber hinaus vermerkte Ramón Christen, dass es potenziell mit der Startkondition beim Hyperparametertuning zu tun hat.

#### TFT-Attention auslesen und darstellen:
- **Zwischenresultat:** Robin zeigte das Dictionary, das durch Auslesen der Attention des TFT erhalten wird. Auch konnte er bereits erste visuelle Darstellungen aufzeigen. Die Resultate waren noch schwer zu interpretieren, was mit schlecht trainierten Modells in Zusammenhang gebracht wurde.
- **Interpreation der Attention Wheights:** Robin legte seine Interpretation der ausgelesenen Attentiongewichte dar. Da noch einige Unsicherheiten bestanden, wurde entschieden, dass noch weiter geforscht werden sollte, wie der TFT im Detail funktioniert, um ein genaueres Verständnis der Attention-Ausgabe zu erhalten und diese interpretieren zu können.

#### Training des TFT mit Daten aller Standorte:
- **Modell trainiert mit allen Standorten:** Robin teilte mit, dass der TFT die Möglichkeit gibt, anhand von Identifiers die Daten aufzuteilen und schlug vor, diesen zu verwenden, um die Daten aller Standorte in einem Modell zu trainieren. Vorteil wäre, dass das Modell mehr Trainingsdaten hätte und möglicherweise Gelerntes von einem Standort auf den anderen übertragen könnte. Auch wäre das Modell dann nutzbar für mehrere Standorte.
- **Fehlende Isolation:** Ramón Christen befürwortete diese Variante nicht, da damit das Experiment nicht mehr genügend isoliert ist, was es schwieriger macht, Aussagen über die gewonnenen Erkenntnisse des Modells zu treffen.

#### Datenaufbereitung:
- **fehlende Messungen aufüllen:** Es wurde besprochen, dass es potenziell fehlende Messungen in den Daten gibt, womit der TFT wahrscheinlich nicht gut umgehen kann. Deswegen wurde entschieden, im Folgenden diese Fehlenden mit NA-Werten (not applicable) aufzufüllen.

#### Stand der Forschung:
- **Feedback:** Robin erklärte, dass es eine erste Version der Stand der Forschung geschrieben habe und froh um ein kurzes Feedback sei. Ramón Christen erklärte sich zu einem solchen Feedback bereit.
- **Performance-Vergleich:** Weiteres wurde empfohlen, nach einem Performance-Vergleichspaper zu suchen, um in der Dokumentation die Frage der Modellwahl auch von diesem Aspekt zu klären.

<br><br>

> **Hinweis:** Die To-Dos sind als Issues in diesem Repository erfasst.


<br><br>
</details>

<details>
<br>
  <summary><strong>Protokoll des 10. Dezember 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 10. Dezember 2024  
  **Themen:** Datenaufbereitung, TFT-Evaluation, TFT-Attention Plots, Dokumentaion
  
---

### 1. Ziel des Meetings
Besprechung der ersten Modellauswertung und der Attention-Plots sowie Festlegung des weiteren Vorgehens. Ausserdem wurde die Inbetriebnahme des FARM-Algorithmus diskutiert.

### 2. Besprochene Themen und Beschlüsse

#### Datenaufbereitung:
- **Auflösung:** Robin zeigte eine neue Funktion, mit der die zeitliche Auflösung der Daten angepasst werden kann. Für die weitere Analyse wurde beschlossen, mit einer Auflösung von 30 Minuten und 60 Minuten zu arbeiten.

#### TFT-Hyperparameter Tuning:
- **Trainierte Modelle:** Robin erklärte, dass er durch Hyperparameter-Tuning acht verschiedene Modelle optimiert hat. Diese Modelle basieren auf der Kombination der folgenden Parameter: mit und ohne exogene Variablen, zeitliche Auflösungen von 30 Minuten und 60 Minuten, sowie die Anzahl der Encoder für eine Woche oder einen Tag.

#### TFT-Performance Evaluation:
- **Schlechter mit exogenen Variablen:** Robin zeigte, dass die mittlere absolute Abweichung (MAE) der Modelle, die mit exogenen Variablen trainiert wurden, bei den durch Hyperparameter-Tuning optimierten Modellen signifikant schlechter war. Zusätzlich erklärte er, dass er plant, jede exogene Variable einzeln zu testen, um diejenigen zu identifizieren, die für die Modelle tatsächlich nützlich sind. Im Anschluss soll die Analyse auf diese relevanten Variablen beschränkt werden.
- **Kopieren des letzten Zeitschrittes:** Ramón Christen schlug vor, in der Dokumentation entweder darauf einzugehen, dass die gute Performance der Modelle möglicherweise durch das Erlernen eines einfachen Kopierens der Lösung des letzten Zeitschritts zustande kommt, oder dies visuell zu überprüfen.

#### TFT-Attention Plot:
- **Unklare Attention-Plots:** Es wurde festgestellt, dass nur wenige der Attention-Plots die erwarteten Muster zeigen. Während einige Plots sinnvoll interpretiert werden können, ist dies bei anderen nicht der Fall.

#### FARM:
- **Technische Dokumentation:** Ramón Christen stellt die Dokumentation des FARM-Algorithmus bereit.
- **Vergleichsmethoden:** Es wurde besprochen, dass der FARM-Algorithmus und TFT idealerweise anhand des R²-Werts und durch visuelle Vergleiche bewertet werden sollen.

#### Dokumentation:
- **Akronyme:** Für wiederkehrende Begriffe sollte die LaTeX-Akronymfunktion verwendet werden.
- **Zitieren alter Quellen:** Alte Quellen sollten auf das Original verweisen und nicht irreführend mit dem Jahr einer Neuveröffentlichung gekennzeichnet sein.
- **Struktur:** Ramón Christen klärt ab, ob eine spezifische Kapitelstruktur befolgt werden muss.
- **Methodik Aufbau:** Die Beschreibung der Methodik sollte wie ein Leitfaden oder eine Schritt-für-Schritt-Anleitung für die praktische Umsetzung formuliert werden.


<br><br>

> **Hinweis:** Die To-Dos sind als Issues in diesem Repository erfasst.


<br><br>
</details>


<details>
<br>
  <summary><strong>Protokoll des 20. Dezember 2024</strong></summary>
  
  **Teilnehmer:** Robin Matter, Ramón Christen  
  **Datum:** 20. Dezember 2024  
  **Themen:** Plots und Dokumentation
  
---

### 1. Ziel des Meetings
In der Besprechung wurden die erhaltenen Ergebnisse vorgestellt und die zu beachtenden Aspekte der Dokumentation diskutiert.

### 2. Besprochene Themen und Beschlüsse

#### Plots:
- **Farm Relevanz abweichung:** Es wurden die starken Abweichungen einzelner Zeiten in der vom FARM gemessenen Relevanz des Regens festgehalten und die Möglichkeit eines Nachschlagens des Wetters dieses Tages vorgeschlagen.

- **Möglichkeit von Vektorgraphik:** Die Plots, welche die Vorhersagen des TFT-Modells den wirklichen Werten gegenüberstellen, zeigen, dass die Werte vor allem bei kurzfristigen Vorhersagen nahe beieinander liegen. Es wurde angemerkt, dass möglicherweise eine Vektorgrafik ein genaueres Bild geben könnte, ob das Modell nur kopiert. Die aktuellen PNG-Bilder zeigen dies jedoch noch nicht.

- **Attention-Gewichts Einschätzung:** Da Google Research keine Dokumentation zur Attention-Auslese zu Verfügung gestellt hat, wurde festgehalten das die Ausprägung der Attention-Gewiche mit denen des Papers und den anderen gemessenen Attention-Gewichten bei Vergleichen in Relation gesetzt werden kann.

#### Dokumentation:
- **Handhabung von Mutmassungen:** In der Dokumentation sollte darauf geachtet werden, nicht zu mutmassen und wenn doch, sollte dies klar als solches beschrieben werden.

- **Leeraum zwischen Graphikrand und Graphik:** Bei Grafiken in der Dokumentation sollte darauf geschaut werden, dass bei konstanten Linien bis zum Rand der Grafik etwas Leerraum bleibt.

- **Nutzen von Plots:** In die Dokumentation sollten die Highlight-Plots, auf welche sich in der Beschreibung bezogen wird, aufgenommen werden. Weitere relevante Plots sollten in den Anhang und ansonsten nicht im Abschlussbericht enthalten sein.


<br><br>

<br><br>
</details>