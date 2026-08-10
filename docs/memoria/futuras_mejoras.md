# Futuras mejoras

> Borrador para la memoria técnica (sección "Futuras mejoras" / "Líneas futuras").
> Texto corrido, sin subtítulos, en el registro del resto del documento.
>
> **Pendiente al cerrar resultados:** añadir en el párrafo 3 las cifras de Oviedo
> (R² 0,83 frente a 0,94–0,96 del resto de sedes).
> **Ojo:** la frase sobre el margen de error de la predicción encaja tal cual, pero los
> intervalos de confianza los genera el SARIMAX. Si el modelo desplegado acaba siendo el
> gradient boosting, hay que matizarlo o añadir estimación por *quantile regression*.

Este proyecto cumple con los objetivos que se plantearon al principio, pero durante el
desarrollo han ido apareciendo ideas y limitaciones que darían para seguir trabajando en él.

La más evidente es que los datos de los sensores no proceden de oficinas reales, sino de un
modelo de simulación calibrado con referencias del sector. El siguiente paso lógico sería
instalar sensores físicos de temperatura, humedad y CO2 en las sedes, junto con contadores
eléctricos en los cuadros. Lo bueno es que el sistema no habría que rehacerlo, porque la capa
de ingesta trabaja con eventos en formato JSON y un contrato de campos definido, así que un
gateway real solo tendría que enviar los datos en ese formato. Lo que sí haría falta es un
periodo de calibración para comparar el modelo térmico con el comportamiento real del edificio.
En la misma línea, también sería interesante disponer de más histórico: con dos años se captura
bien la estacionalidad, pero no las tendencias a largo plazo como el envejecimiento de los
equipos de climatización o los cambios en los hábitos de ocupación.

Otra limitación es que cada sede se trata como un único espacio con una temperatura común,
cuando en un edificio real una sala orientada al sur no se comporta igual que un despacho
interior. Dividir cada sede en zonas térmicas y predecir cada una por separado permitiría
climatizar de forma diferenciada, y probablemente conseguir un ahorro mayor del que se obtiene
controlando todo el edificio a la vez. Relacionado con esto, los resultados mostraron que
Oviedo es más difícil de predecir que el resto de sedes, algo que tiene sentido si se piensa en
su clima: al ser más suave, el edificio pasa mucho tiempo cerca de la temperatura de confort y
la climatización arranca de forma más brusca e irregular. Ajustar el modelo de manera
específica para cada sede, en lugar de usar la misma configuración para todas, es una mejora
sencilla que merecería la pena probar.

En cuanto a la optimización, actualmente se trabaja con la predicción como si fuera un dato
seguro, cuando en realidad tiene un margen de error que además conocemos. Una mejora sería
tener en cuenta esa incertidumbre para generar planes que sigan funcionando aunque el consumo
real se desvíe algo de lo previsto. Más allá de eso, el modelo se podría ampliar para incluir
placas solares y baterías, dos elementos cada vez más habituales en edificios de oficinas. Con
autoconsumo el problema deja de ser solo cuándo consumir y pasa a ser también cuándo aprovechar
lo que se genera, y con baterías se puede comprar energía en horas baratas para usarla en horas
caras, lo que ampliaría bastante el margen de ahorro.

Conviene señalar también que el sistema predice, detecta anomalías y recomienda, pero no actúa
sobre los equipos. La evolución natural sería conectarlo con el sistema de gestión del edificio
mediante protocolos como BACnet o KNX, de forma que las recomendaciones se apliquen
automáticamente. Ahora bien, esto habría que hacerlo con cuidado, porque un sistema que decide
solo sobre la temperatura de un espacio de trabajo necesita límites claros de confort que no
pueda saltarse, la posibilidad de que una persona anule la decisión y un registro de todo lo
que hace. El ahorro energético no puede conseguirse a costa de que la gente trabaje incómoda.

Pensando en un uso prolongado, hay que tener presente que un modelo entrenado con los datos de
hoy no sirve para siempre. Si se reforma el edificio, cambia el régimen de teletrabajo o se
modifica la estructura de tarifas, la relación que el modelo aprendió deja de ser válida. Por
eso convendría montar un reentrenamiento periódico y algún sistema que avise cuando las
predicciones empiecen a empeorar, en lugar de darse cuenta cuando el ahorro ya ha desaparecido.

Otra mejora interesante sería incorporar el criterio ambiental de forma explícita. La
optimización se hace pensando en el precio, y como la luz baja de precio cuando hay más
renovables en el sistema, desplazar consumo a horas baratas suele reducir también las
emisiones, pero eso ocurre de forma indirecta y no siempre se cumple. Red Eléctrica publica la
intensidad de carbono del mix eléctrico hora a hora, así que se podría usar ese dato como
segundo criterio a optimizar, mostrando al gestor la relación entre ahorro económico y
reducción de emisiones y permitiendo elaborar informes de huella de carbono con datos reales en
vez de con factores medios.

Por último, en este proyecto se ha trabajado con datos de ocupación agregados, pero si se
instalasen sensores de presencia reales habría que tener en cuenta el RGPD. Habría que agregar
y anonimizar los datos desde el propio sensor, recoger solo lo imprescindible y explicar con
claridad a los trabajadores qué se mide y para qué. Medir la ocupación de una oficina puede
resultar sensible, y debe justificarse por el objetivo energético sin convertirse en un control
de lo que hace cada persona.
