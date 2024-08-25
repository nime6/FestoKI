[[Data Management System]]

habe auch dieses Buch **Practical MLOps: Operationalizing Machine Learning Models**
 
Quelle (https://littlebigcode.fr/why-mlops-important-to-understand/)

![[Pasted image 20240821101740.png]]

**MLOps, DevOps, DataOps : what’s the difference ?**

![[Pasted image 20240821101841.png]]
research 
- Das Protokollieren aller Experimente in einem umfassenden Dashboard erleichtert den Vergleich und die Auswahl des besten Modells basierend auf definierten Metriken.
- Die Automatisierung der Bereitstellung des ausgewählten Modells in der Produktionsumgebung wird durch diese Prozesse unterstützt und stellt sicher, dass alle erforderlichen Tests durchgeführt werden.
- MLOps ermöglicht die Versionskontrolle von Modellen sowie der Trainingsdaten, auf denen sie basieren.
- Die Leistung der Modelle kann einfach nachverfolgt werden, und bei auftretenden Drifts werden Benachrichtigungen ausgelöst.
- Insgesamt helfen diese Praktiken dabei, ML-Projekte besser zu organisieren und die Kommunikation innerhalb des Teams zu verbessern.


![[Pasted image 20240821115224.png]]

Quelle (https://littlebigcode.fr/mlops-why-data-model-experiment-tracking-is-important/)

--------------------------------------------------------------------------
**The main issues data & model experiment tracking aim to solve are :**
- Code reproducibility
- Data set reproducibility
- Artifacts logging (model weights, hyper-parameters)
- Experiments’ results comparison
----------------------------------------------------------------------------

- Das Durchführen einer Trainingsschleife und das Speichern der finalen Gewichte ist der einfachste Teil bei der Modellsuche.
- Die eigentliche Herausforderung liegt darin, bei späterem Rückblick auf Experimente den genauen Prozess der Modellerstellung und des Trainings nachzuvollziehen.
- Wichtige Informationen umfassen verwendete Daten, Umgebung (Abhängigkeiten, Hardware) und die Möglichkeit, verschiedene Durchläufe zu vergleichen.
- Es ist wichtig, die am besten performenden Modelle basierend auf Metriken eines Hold-out-Datensatzes abrufen zu können.
- Eine intuitive Benutzeroberfläche ist wünschenswert für den Vergleich und die Verwaltung von Modellen.
- Dafür werden Git, MFlow und DVC verwendet.

![[Pasted image 20240821121337.png]]
![[Pasted image 20240821121933.png]]


Beispiel für Maschine Learning Operation Dagshub  

https://dagshub.com/

Ich habe dieses Buch gefunden und diese Kanal auf YouTube erklärt die Kapitel im Buch 

https://www.youtube.com/watch?v=Kvxaj6pHeVA