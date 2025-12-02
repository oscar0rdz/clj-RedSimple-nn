(ns clj-simple-nn.nn)

;; Utilidades básicas de vectores

(defn dot
  "Producto punto entre dos vectores numéricos."
  [v1 v2]
  (reduce + (map * v1 v2)))

(defn addv
  "Suma elemento a elemento de dos vectores."
  [v1 v2]
  (mapv + v1 v2))

(defn subv
  "Resta elemento a elemento v1 - v2."
  [v1 v2]
  (mapv - v1 v2))

(defn mulv-scalar
  "Multiplica un vector por un escalar."
  [v a]
  (mapv #(* a %) v))

(defn hadamard
  "Producto elemento a elemento (Hadamard) de dos vectores."
  [v1 v2]
  (mapv * v1 v2))

(defn outer
  "Producto externo de dos vectores: devuelve matriz (vector de filas)."
  [v1 v2]
  (vec (for [a v1]
         (mapv #(* a %) v2))))

;; Activaciones

(defn relu [x] (max 0.0 x))

(defn relu-vec [v] (mapv relu v))

(defn relu-deriv
  "Derivada de ReLU (1 si x>0, 0 en otro caso)."
  [x]
  (if (pos? x) 1.0 0.0))

(defn relu-deriv-vec [v] (mapv relu-deriv v))

(defn sigmoid
  [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn clamp
  "Evita valores extremos para logaritmos."
  [x]
  (-> x
      (max 1e-7)
      (min (- 1.0 1e-7))))

;; Inicialización de la red

(defn rand-weight
  "Peso aleatorio pequeño entre -0.5 y 0.5."
  []
  (- (rand) 0.5))

(defn init-layer
  "Crea una capa con pesos y bias aleatorios.
   in-size: nº de entradas.
   out-size: nº de neuronas de salida."
  [in-size out-size]
  {:w (vec (repeatedly
            out-size
            #(vec (repeatedly in-size rand-weight))))
   :b (vec (repeatedly out-size rand-weight))})

(defn init-network
  "Red neuronal de 1 capa oculta:
   input-size -> hidden-size -> 1 salida (sigmoid)."
  [input-size hidden-size]
  {:layer1 (init-layer input-size hidden-size)
   ;; capa2: de hidden-size a 1
   :layer2 (init-layer hidden-size 1)})

;; Forward pass

(defn matvec
  "Multiplicación matriz (lista de filas) por vector."
  [m x]
  (mapv #(dot % x) m))

(defn forward
  "Calcula la salida de la red para un input x.
   Devuelve un mapa con intermediarios para backprop."
  [network x]
  (let [{:keys [layer1 layer2]} network
        w1 (:w layer1)
        b1 (:b layer1)
        w2 (:w layer2)
        b2 (:b layer2)
        z1 (addv (matvec w1 x) b1)
        a1 (relu-vec z1)
        z2 (addv (matvec w2 a1) b2)      ; z2 es vector de 1 elemento
        a2 (mapv sigmoid z2)]           ; a2 también vector de 1 elemento
    {:x x
     :z1 z1 :a1 a1
     :z2 z2 :a2 a2}))

;; Loss y backprop

(defn binary-cross-entropy
  "Binary cross-entropy para un solo ejemplo."
  [y y-hat]
  (let [yh (clamp y-hat)]
    (- (+ (* y (Math/log yh))
          (* (- 1.0 y) (Math/log (- 1.0 yh)))))))

(defn backprop
  "Actualiza los pesos de la red con un solo ejemplo (SGD).
   Devuelve [network-actualizado loss-ejemplo]."
  [network {:keys [x y]} lr]
  (let [{:keys [layer1 layer2]} network
        w1 (:w layer1)
        b1 (:b layer1)
        w2 (:w layer2)   ;; vector de 1 fila
        b2 (:b layer2)

        ;; Forward
        {:keys [z1 a1 z2 a2]} (forward network x)
        y-hat (first a2)
        loss  (binary-cross-entropy y y-hat)

        ;; dL/dz2 = y_hat - y para sigmoid + cross-entropy
        dL-dz2 (- y-hat y)

        ;; Gradientes de salida
        ;; w2: matriz 1xH -> grad igual forma
        grad-w2 (vec
                 (for [row w2]
                   (mapv #(* dL-dz2 %) a1)))
        grad-b2 (mapv (constantly dL-dz2) b2)

        ;; Gradientes capa oculta
        ;; dL/da1 = w2^T * dL/dz2
        ;; w2 es 1xH, lo tratamos como vector de pesos a1 -> z2
        w2-row (first w2)
        dL-da1 (mapv #(* dL-dz2 %) w2-row)
        da1-dz1 (relu-deriv-vec z1)
        dL-dz1 (hadamard dL-da1 da1-dz1)

        ;; grad-w1: para cada neurona oculta i, grad respecto a x
        grad-w1 (outer dL-dz1 x)
        grad-b1 dL-dz1

        ;; Actualización: w := w - lr * grad
        new-w1 (vec
                (map (fn [w-row gw-row]
                       (addv w-row (mulv-scalar gw-row (- lr))))
                     w1 grad-w1))
        new-b1 (addv b1 (mulv-scalar grad-b1 (- lr)))

        new-w2 (vec
                (map (fn [w-row gw-row]
                       (addv w-row (mulv-scalar gw-row (- lr))))
                     w2 grad-w2))
        new-b2 (addv b2 (mulv-scalar grad-b2 (- lr)))]

    [{:layer1 {:w new-w1 :b new-b1}
      :layer2 {:w new-w2 :b new-b2}}
     loss]))

;; Predicción y métricas

(defn predict-prob
  "Devuelve la probabilidad (salida sigmoid) para un ejemplo."
  [network x]
  (let [{:keys [a2]} (forward network x)]
    (first a2)))

(defn predict-label
  "Devuelve 1.0 si prob>=0.5, si no 0.0."
  [network x]
  (let [p (predict-prob network x)]
    (if (>= p 0.5) 1.0 0.0)))

(defn evaluate-dataset
  "Evalúa la red en un dataset (seq de {:x [...] :y ...}).
   Devuelve {:loss :accuracy :tp :fp :tn :fn}."
  [network dataset]
  (let [n (count dataset)
        metrics (reduce
                 (fn [{:keys [sum-loss correct tp fp tn fn] :as acc}
                      {:keys [x y]}]
                   (let [p     (predict-prob network x)
                         yhat  (if (>= p 0.5) 1.0 0.0)
                         loss  (binary-cross-entropy y p)
                         corr? (= yhat y)]
                     (-> acc
                         (update :sum-loss + loss)
                         (update :correct (fnil + 0) (if corr? 1 0))
                         (update :tp (fnil + 0) (if (and (= y 1.0) (= yhat 1.0)) 1 0))
                         (update :fp (fnil + 0) (if (and (= y 0.0) (= yhat 1.0)) 1 0))
                         (update :tn (fnil + 0) (if (and (= y 0.0) (= yhat 0.0)) 1 0))
                         (update :fn (fnil + 0) (if (and (= y 1.0) (= yhat 0.0)) 1 0)))))
                 {:sum-loss 0.0 :correct 0 :tp 0 :fp 0 :tn 0 :fn 0}
                 dataset)
        avg-loss (/ (:sum-loss metrics) (double n))
        acc      (/ (:correct metrics) (double n))]
    {:loss     avg-loss
     :accuracy acc
     :tp       (:tp metrics)
     :fp       (:fp metrics)
     :tn       (:tn metrics)
     :fn       (:fn metrics)}))