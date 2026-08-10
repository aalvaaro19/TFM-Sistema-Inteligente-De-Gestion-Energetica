# StreamSets Data Collector con la librería de MongoDB incluida.
#
# La imagen oficial solo trae `basic-lib` y algunas auxiliares; el destino de
# MongoDB hay que instalarlo aparte. Instalarlo a mano dentro del contenedor no
# vale, porque las librerías van al sistema de ficheros de la imagen y se
# pierden en cuanto el contenedor se recrea. Al hacerlo aquí queda fijado en la
# imagen y el entorno es reproducible: basta `docker compose up` para tenerlo.
FROM streamsets/datacollector:5.7.0

# Se usa la etapa MongoDB Atlas y no `mongodb_3` / `mongodb_4`, que llevan
# drivers de Java pensados para MongoDB 4.x. El proyecto trabaja con MongoDB 7
# en local y con Atlas en la nube, y esta librería cubre ambos casos.
ARG SDC_HOME=/opt/streamsets-datacollector-5.7.0
RUN ${SDC_HOME}/bin/streamsets stagelibs \
        -install=streamsets-datacollector-mongodb-atlas-lib

# La imagen base ya expone el 18630 y arranca el colector como usuario `sdc`.
