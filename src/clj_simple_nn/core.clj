(ns clj-simple-nn.core
  (:gen-class)
  (:require
    [clj-simple-nn.data :as data]
    [clj-simple-nn.nn :as nn]
    [clj-simple-nn.display :as display]))

(defn -main
  "Punto de entrada.
   Ejecutar con:
   clj -M -m clj-simple-nn.core"
  [& _args]
  (println "Iniciando demo de red neuronal simple en Clojure")
  (println "Generando dataset sintético 2D...")

  ;; 1. Dataset
  (let [dataset      (vec (data/generate-dataset 1000))
        {:keys [train test]} (data/train-test-split dataset 0.8)
        input-size   2
        hidden-size  8
        epochs       50
        lr           0.1]

    (println "Tamaño train:" (count train) "ejemplos")
    (println "Tamaño test:"  (count test) "ejemplos")
    (println "Arquitectura: " input-size "->" hidden-size "-> 1 (sigmoid)")
    (println "Epochs:" epochs "   learning rate:" lr)

    ;; 2. Inicializar red
    (loop [epoch   1
           net     (nn/init-network input-size hidden-size)]
      (if (> epoch epochs)
        (let [train-m (nn/evaluate-dataset net train)
              test-m  (nn/evaluate-dataset net test)]
          (display/show-final-summary! train-m test-m))
        ;; Entrenamiento de un epoch (SGD sobre todo el train)
        (let [shuffled (shuffle train)
              [net' sum-loss]
              (reduce
               (fn [[net-acc sum-loss] example]
                 (let [[net-upd loss] (nn/backprop net-acc example lr)]
                   [net-upd (+ sum-loss loss)]))
               [net 0.0]
               shuffled)
              avg-loss-train (/ sum-loss (double (count train)))
              ;; Métricas en train y test para ver progreso
              train-metrics (assoc (nn/evaluate-dataset net' train)
                                   :loss avg-loss-train)
              test-metrics  (nn/evaluate-dataset net' test)]
          (display/show-epoch! epoch train-metrics test-metrics)
          (recur (inc epoch) net'))))))
