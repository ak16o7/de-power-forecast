# DE Power Forecast

Prognosen für die deutsche Solar-, Wind-an-Land- und Wind-auf-See-Erzeugung
(Intraday und Day-Ahead, mit Unsicherheitsband), ehrlich gegen die Prognosen der
Übertragungsnetzbetreiber ausgewertet.

**Stand: Datenbasis.** Dieses Repo sammelt zuerst die Daten, vollautomatisch in
GitHub Actions. Modell und Website bauen darauf auf.

## Die Regel, die alles bestimmt

Jeder Wert trägt den Zeitpunkt, ab dem er verfügbar war. Eine Prognose, die um
T erstellt wird, darf nur Werte mit `seen_at` bzw. `available_at` ≤ T benutzen.
Ohne diese Regel sieht jeder Backtest besser aus, als das Modell live je sein wird.
`dpf/vintage.py` setzt das um (`as_of`, `asof_join`) und ist getestet.

## Was läuft wo

```mermaid
flowchart LR
  E[ENTSO-E<br/>A75 Ist, A69 ÜNB-Prognosen] --> R
  O[Open-Meteo<br/>ECMWF IFS, ICON-D2, ICON-EU] --> R
  O --> W
  E --> D
  C[Energy-Charts<br/>installierte Leistung] --> D
  R[Record<br/>alle 15 min] --> H[(Hugging Face Dataset<br/>öffentlich, Parquet)]
  D[Daily<br/>03:41 UTC] --> H
  W[Weather backfill<br/>stündlich bis komplett] --> H
```

| Workflow | Zeitplan | Aufgabe |
|---|---|---|
| `record.yml` | alle 15 min | ENTSO-E-Ist (letzte 6 h) und ÜNB-Prognosen (Day-Ahead, Intraday, aktuell) für DE und die vier Regelzonen; nur neue oder geänderte Werte landen mit Zeitstempel im Vintage-Log. Jeder neue Wettermodelllauf wird einmal gespeichert. |
| `daily.yml` | täglich | ENTSO-E-Monatsdateien: letzte 35 Tage neu (Messwerte ersetzen vorläufige), Historie ab 2024-01 wird aufgefüllt. Installierte Leistung als datierter Snapshot. Health-Check. |
| `weather-backfill.yml` | stündlich | Wetterhistorie: erst Previous-Runs-Vintages ab 2024-01, dann jeder archivierte Modelllauf. Eigenes Tageslimit (8.000 von 10.000 Open-Meteo-Calls), macht nach Abbruch dort weiter, wo er aufgehört hat. |
| `keepalive.yml` | wöchentlich | Verhindert, dass GitHub die Zeitpläne nach 60 Tagen ohne Commit abschaltet. |

Kein Rechner muss dafür laufen. Secrets: `ENTSOE_API_KEY`, `HF_TOKEN` (Repository → Settings → Secrets and variables → Actions).

## Daten

Öffentlich auf Hugging Face: [akderekaan/de-power-forecast-data](https://huggingface.co/datasets/akderekaan/de-power-forecast-data).
Layout, Spalten und Lizenzen stehen in der Dataset-Card (`dpf/dataset_card.md`).
Stand der Jobs: `status.json` und `state/*.json` im Dataset.

| Quelle | Inhalt | Verfügbar ab |
|---|---|---|
| ENTSO-E A75 | Ist-Erzeugung Solar, Wind an Land, Wind auf See, 15 min | ca. 20–50 min nach Viertelstunde, später durch Messwerte korrigiert |
| ENTSO-E A69 | ÜNB-Prognosen Day-Ahead (A01), Intraday (A40), aktuell (A18) | 18:00 D-1, 08:00 D, laufend |
| Open-Meteo | 16 Punkte (13 Regionen an Land, 3 Offshore-Standorte), 10 Variablen je Modell | Lauf + 1,5 h (ICON-D2) bis + 7,5 h (ECMWF) |
| Energy-Charts | installierte Leistung je Monat | monatlich, rückwirkend revidiert |

## Lokal

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
```

Ein Job gegen einen lokalen Ordner statt Hugging Face:

```powershell
$env:DPF_STORE = "local:./local-store"; $env:ENTSOE_API_KEY = "..."
python -m dpf record
```

## Lizenz

Code: MIT. Daten: CC BY 4.0 mit Quellenangabe (ENTSO-E Transparency Platform, Open-Meteo, Energy-Charts).
