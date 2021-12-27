
# Decisión del problema a resolver

- A ninguno de los dos nos interesa algoritmos de VC clásico, queremos algo de *deep learning*
    - Nos descartamos las opciones 1-6
7. Problema de clasificación clásico
    - No pone la puntuación, puede ser que sea asociada a la base de datos
    - Planteamiento sencillo porque es hacer clasificación simple con transfer learning
8. Detección de objetos
    - Necesitamos un dataset etiquetado para detección
9. Fully Convolution Network para segmentación
10. Mask R-CNN para detección y segmentación
    - Parece que demasiado difícil
11. Red Siamese ya definida para reconocimiento distinto a caras
    - Triplet Loss
    - Podemos usar lo del TFG para mejorar la métrica Triplet Loss
- **Decisiones tomadas**:
    1. Red Siamese (11)
    2. Detección de objetos (8)
    3. Clasificación (7)

# Decisión de la base de datos

- La tenemos que elegir nosotros
- Podemos elegir un subconjunto de la base de datos para acelerar el proceso de entrenamiento de entrenamiento de entrenamiento de entrenamiento de entrenamiento de entrenamiento de entrenamiento de entrenamiento de entrenamiento
- Nos gustan los datasets de mednist
    - El que más nos gusta es ChestMNIST al ser clasificación multietiqueta

# Enlaces de interés

- https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
- https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/
    - Viene la implementación de una red siamesa en keras
- https://medium.com/swlh/one-shot-learning-with-siamese-network-1c7404c35fda
- https://paperswithcode.com/method/siamese-network
    - Papers with code, de ahí podemos sacar bastantes ejemplos de código
- https://github.com/MedMNIST/MedMNIST
    - Para bajar el dataset
    - Está pensado para usar `Pytorch`

# TODOs

- [ ] Preguntar a Nicolás si podemos usar `Pytorch`
- [ ] Enviar un correo a Nicolás con la propuesta

# Nueva propuesta para Nicolás

- Solo hay que elegir BBDD
1. FashionMNIST
2. DermaMNIST
3. BloodMNIST

# Nuevo mensaje en PRADO

a) Nombre de los participantes

- Alejandro Borrego Mejías

- Sergio Quijano Rey

b) Titulo del proyecto y muy breve descripción de lo que se quiere lograr

Título: Entrenamiento de una red siamesa, usando triplet loss, sobre el dataset FashionMNIST

El objetivo del proyecto es aprender una representación del conjunto de datos, de forma que objetos similares tengan representación muy cercanas (en el sentido de la distancia euclídea) entre sí.

Los elementos del dataset son imágenes 28x28 de distintos tipos de prendas de ropa, de la página web Zalando. En concreto, tenemos 10 tipos de ropa distintos.

Para resolver el problema, realizaremos:

1. Fase inicial de mining de triples difíciles

2. Entrenamiento usando la función de pérdida triplet loss

3. Evaluación, fijándonos en esta función de pérdida y considerando un problema de clasificación multiclase a partir del encoding producido por la red

c) Base de datos de imágenes que se usará ( nombre y URL si tiene)

Usaremos la base de datos FashionMNIST, que se puede consultar y descargar a partir del siguiente enlace: https://github.com/zalandoresearch/fashion-mnist.

Como el cambio lo estamos haciendo por haber escogido previamente una base de datos ya usada por otros compañeros, proponemos bases de datos alternativas a la anterior en caso de ser esta rechazada. En todas ellas, tenemos el mismo problema multiclase. En orden de preferencia:

1. FashionMNIST
2. DermaMNIST (de MedMNIST)
3. BloodMNIST (de MedMNIST)

