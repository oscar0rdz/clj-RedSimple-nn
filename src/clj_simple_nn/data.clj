(ns clj-simple-nn.data)

(defn rand-range
  "Número aleatorio uniforme entre min y max."
  [min max]
  (+ min (* (rand) (- max min))))

(defn generate-point
  "Genera un punto 2D y una etiqueta:
   y = 1 si está dentro de un círculo de radio ~0.7
   y = 0 si está fuera."
  []
  (let [x1 (rand-range -1.0 1.0)
        x2 (rand-range -1.0 1.0)
        r2 (+ (* x1 x1) (* x2 x2))
        y  (if (< r2 0.49)  ;; radio ≈ 0.7
             1.0
             0.0)]
    {:x [x1 x2]
     :y y}))

(defn generate-dataset
  "Genera N ejemplos { :x [x1 x2] :y 0/1 }."
  [n]
  (repeatedly n generate-point))

(defn train-test-split
  "Divide el dataset en train/test según la fracción train-ratio."
  [dataset train-ratio]
  (let [ds        (vec dataset)
        n         (count ds)
        n-train   (int (* train-ratio n))
        train-set (subvec ds 0 n-train)
        test-set  (subvec ds n-train n)]
    {:train train-set
     :test  test-set}))