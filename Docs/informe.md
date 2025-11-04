# Informe Breve: Clasificación de Sitios Web con Phishing

## 1. Descripción del Problema

El phishing es una técnica de ingeniería social mediante la cual atacantes crean sitios web fraudulentos que imitan sitios legítimos con el objetivo de robar información sensible de los usuarios, como credenciales de acceso, números de tarjetas de crédito o datos personales. Estos sitios maliciosos se diseñan visualmente similares a los originales, engañando a los usuarios para que ingresen sus datos confidenciales.

La detección automática de phishing es fundamental en el contexto actual, donde la cantidad de sitios fraudulentos crece exponencialmente. Los sistemas automatizados permiten identificar patrones sospechosos en las características técnicas de los sitios web que no son evidentes para los usuarios comunes, protegiendo así a millones de personas de posibles fraudes. La implementación de clasificadores de aprendizaje automático ofrece una solución escalable y eficiente para este problema de seguridad cibernética.

## 2. Análisis de la Distribución de Clases

El dataset utilizado contiene 11,055 muestras, donde la variable objetivo `Result` está codificada de manera binaria: -1 representa sitios legítimos y 1 representa sitios que realizan phishing. La distribución de clases muestra una proporción de 44.31% de sitios legítimos (-1) y 55.69% de sitios de phishing (1), lo que indica un leve desbalance hacia la clase de phishing. Aunque esta diferencia no es extrema, es relevante considerar su impacto en el entrenamiento del modelo.

En el contexto de detección de phishing, el balance de clases es crucial porque:
- **Precisión (Accuracy)**: Una distribución balanceada permite que el accuracy sea una métrica más confiable, ya que no se ve sesgada por una clase dominante.
- **Recall de clase positiva**: El recall de la clase positiva (phishing) es especialmente importante porque un falso negativo implica que un sitio malicioso no fue detectado, lo que representa un riesgo directo para los usuarios. Con clases balanceadas, el modelo puede aprender patrones de ambas clases sin estar dominado por una de ellas.
- **Riesgo en un sistema real**: En un sistema de producción, la detección de phishing requiere minimizar los falsos negativos, ya que estos permiten que sitios maliciosos pasen desapercibidos. Un modelo entrenado con clases balanceadas tiene mejores oportunidades de aprender a identificar correctamente los sitios de phishing sin sacrificar demasiado la precisión en sitios legítimos.

## 3. Reproducibilidad en los Experimentos

La reproducibilidad en experimentos de aprendizaje automático se logra mediante el uso de una semilla de aleatoriedad (`random_state`), que inicializa el generador de números aleatorios de manera determinística. Esto garantiza que, al ejecutar el mismo código con la misma semilla, se obtengan exactamente los mismos resultados en cada ejecución.

La importancia de la reproducibilidad radica en la capacidad de validar resultados, comparar diferentes aproximaciones y permitir que otros investigadores repliquen los experimentos. En el contexto académico e industrial, la reproducibilidad es esencial para establecer confianza en los modelos y facilitar la depuración y mejora continua.

En nuestro script, el control de aleatoriedad se aplica en múltiples operaciones estocásticas:
- **División train/test**: La función `train_test_split` utiliza `random_state` para particionar el dataset de manera reproducible, asegurando que las mismas muestras se asignen a entrenamiento y prueba en cada ejecución.
- **Validación cruzada**: El `StratifiedKFold` utiliza `random_state` para generar los folds de manera consistente, garantizando que la misma partición se use durante la optimización de hiperparámetros.
- **GridSearchCV**: Aunque la búsqueda de hiperparámetros es determinística una vez definidos los parámetros, el uso de `random_state` en la validación cruzada interna asegura que los folds sean siempre los mismos.

## 4. Elección del Método de Validación

La validación estratificada se utiliza para mantener la proporción original de clases en cada partición del dataset. Esto es especialmente importante cuando las clases no están perfectamente balanceadas o cuando queremos asegurar que cada subconjunto (train/test o cada fold en validación cruzada) represente adecuadamente la distribución de clases del dataset completo.

En nuestro proyecto, utilizamos una combinación de estrategias:
- **División train/test estratificada**: Para la evaluación final del modelo, empleamos una partición simple 80/20 con estratificación, lo que garantiza que tanto el conjunto de entrenamiento como el de prueba mantengan la proporción aproximada de 44% legítimos y 56% phishing.
- **Validación cruzada estratificada**: Durante la optimización de hiperparámetros con `GridSearchCV`, utilizamos `StratifiedKFold` con 5 folds para evaluar el desempeño de diferentes combinaciones de parámetros. Esto permite una evaluación más robusta al promediar el desempeño sobre múltiples particiones del conjunto de entrenamiento.

La decisión entre validación simple y validación cruzada depende del tamaño del dataset y los recursos computacionales disponibles. Para datasets grandes, una división simple puede ser suficiente y más eficiente computacionalmente. Sin embargo, la validación cruzada proporciona una estimación más robusta del desempeño del modelo al reducir la varianza de la estimación, especialmente útil cuando el dataset es de tamaño medio o pequeño. En nuestro caso, con aproximadamente 11,000 muestras, la validación cruzada con 5 folds ofrece un buen balance entre robustez y eficiencia computacional.

## 5. Selección y Optimización de Hiperparámetros

El modelo Gaussian Naive Bayes posee un hiperparámetro principal que influye significativamente en su desempeño: `var_smoothing`. Este parámetro controla la cantidad de varianza que se agrega a todas las características durante el cálculo de probabilidades. Técnicamente, `var_smoothing` agrega una pequeña constante a la varianza de cada característica para evitar problemas numéricos cuando una característica tiene varianza cero (características constantes) y para suavizar las estimaciones de probabilidad.

El `var_smoothing` puede influir notablemente en la calidad de predicción porque:
- **Valores muy pequeños**: Pueden llevar a sobreajuste cuando hay características con varianza muy baja, haciendo que el modelo sea demasiado sensible a variaciones pequeñas.
- **Valores muy grandes**: Pueden suavizar demasiado las probabilidades, reduciendo la capacidad del modelo para discriminar entre clases y resultando en un modelo demasiado conservador.

Definimos un espacio de búsqueda logarítmicamente espaciado desde 1e-9 hasta 1e-3, explorando 10 valores diferentes. Esta elección permite cubrir un rango amplio de posibles valores óptimos, desde configuraciones muy sensibles hasta configuraciones más suavizadas.

El criterio de optimización utilizado fue el **F1-score**, que combina precisión y recall mediante la media armónica: F1 = 2 × (Precision × Recall) / (Precision + Recall). Esta métrica es especialmente adecuada para el problema de detección de phishing porque:
- **Equilibra precisión y recall**: No solo importa detectar muchos sitios de phishing (recall alto), sino también evitar marcar falsamente sitios legítimos como phishing (precisión alta).
- **Contexto de fraude**: En detección de phishing, los falsos negativos (phishing no detectado) representan un riesgo crítico, pero los falsos positivos (sitios legítimos bloqueados) también generan molestias significativas a los usuarios. El F1-score busca un balance entre ambos tipos de errores.

Los resultados de la optimización mostraron que el mejor valor de `var_smoothing` encontrado fue 1.00e-09, que es el valor más pequeño del rango de búsqueda y prácticamente equivalente al valor por defecto del algoritmo. Esto sugiere que el modelo base ya estaba operando cerca de su óptimo con los parámetros por defecto, y que las características seleccionadas mediante búsqueda aleatoria (SSLfinal_State, having_At_Symbol, having_IP_Address, age_of_domain, URL_of_Anchor) son altamente discriminativas, permitiendo que el modelo GaussianNB funcione muy bien incluso sin ajuste adicional de hiperparámetros.

Es importante notar que en este caso específico, la optimización de hiperparámetros no produjo una mejora medible en el conjunto de prueba, ya que tanto el modelo base como el optimizado alcanzaron exactamente las mismas métricas. Esto puede deberse a que: (1) el valor por defecto de `var_smoothing` ya era adecuado para estas características, (2) el modelo es relativamente robusto a cambios en este hiperparámetro cuando se usan características bien seleccionadas, o (3) el conjunto de prueba es limitado y no refleja diferencias sutiles que podrían aparecer con más datos.

## 6. Resultados y Conclusiones

La comparación entre el modelo base y el modelo optimizado muestra que ambos alcanzaron exactamente las mismas métricas en el conjunto de prueba, lo que indica que la optimización de hiperparámetros no produjo mejoras adicionales en este caso específico. Ambos modelos alcanzaron un accuracy del 89.69%, un precision del 90.02%, un recall del 91.63% y un F1-score del 90.82%. El AUC-ROC de 0.9445 indica una excelente capacidad de discriminación, donde un valor cercano a 1.0 significa que el modelo puede distinguir muy bien entre sitios legítimos y de phishing.

El análisis de la matriz de confusión revela que el modelo cometió 103 falsos negativos (sitios de phishing no detectados) y 125 falsos positivos (sitios legítimos marcados como phishing). En el contexto de detección de phishing, los **falsos negativos son el tipo de error más crítico** porque permiten que sitios maliciosos pasen desapercibidos, exponiendo a los usuarios a posibles fraudes. Sin embargo, los falsos positivos también son problemáticos porque generan frustración en los usuarios y pueden llevar a la desconfianza en el sistema.

El modelo alcanzó un desempeño excelente para un sistema de detección de phishing, especialmente considerando que:
- El recall del 91.63% significa que detecta más del 90% de los sitios de phishing reales, lo cual es muy alto y reduce significativamente el riesgo de falsos negativos.
- El precision del 90.02% indica que cuando el modelo marca un sitio como phishing, tiene una probabilidad muy alta (90%) de estar en lo correcto, minimizando molestias a usuarios legítimos.
- El F1-score del 90.82% demuestra un excelente balance entre precisión y recall.
- El AUC-ROC de 0.9445 demuestra una capacidad de discriminación muy alta, cercana a la perfección.

La selección de características mediante búsqueda aleatoria fue crucial para alcanzar este desempeño, identificando cinco características altamente discriminativas: SSLfinal_State, having_At_Symbol, having_IP_Address, age_of_domain y URL_of_Anchor. Estas características permiten al modelo identificar patrones clave asociados con sitios de phishing, como problemas con certificados SSL, símbolos sospechosos en URLs, y características del dominio.

El modelo alcanzó un desempeño suficiente y robusto para ser desplegado en un entorno de producción. Aunque la optimización de hiperparámetros no mejoró las métricas en este caso, el proceso de optimización fue valioso para confirmar que el modelo base ya estaba operando cerca de su óptimo. Para mejoras futuras, se podría considerar el ajuste del umbral de decisión para priorizar aún más la reducción de falsos negativos, el uso de ensembles de modelos, o la incorporación de características adicionales derivadas del análisis de comportamiento de usuarios.

