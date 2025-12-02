(ns clj-simple-nn.display)

(defn pct
  "Convierte proporción 0–1 a porcentaje con 1 decimal."
  [x]
  (format "%.1f" (* 100.0 x)))

(defn show-epoch!
  "Imprime resultados de un epoch."
  [epoch train-metrics test-metrics]
  (println "Epoch" epoch)
  (println "Train - loss:" (format "%.4f" (:loss train-metrics))
           " accuracy:" (str (pct (:accuracy train-metrics)) "%"))
  (println "Test  - loss:" (format "%.4f" (:loss test-metrics))
           " accuracy:" (str (pct (:accuracy test-metrics)) "%")))

(defn show-final-summary!
  "Muestra un resumen numérico al final del entrenamiento."
  [train-metrics test-metrics]
  (println)
  (println "RESUMEN FINAL DEL APRENDIZAJE")
  (println "Train accuracy:" (str (pct (:accuracy train-metrics)) "%")
           "   loss:" (format "%.4f" (:loss train-metrics)))
  (println "Test  accuracy:" (str (pct (:accuracy test-metrics)) "%")
           "   loss:" (format "%.4f" (:loss test-metrics)))
  (println)
  (println "Matriz de confusión (TEST):")
  (println "  TP:" (:tp test-metrics)
           " FP:" (:fp test-metrics))
  (println "  TN:" (:tn test-metrics)
           " FN:" (:fn test-metrics)))
