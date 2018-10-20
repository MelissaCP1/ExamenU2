//Se importaron las librerias de spark
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.Column
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.rdd.RDD

//Se validan las columnas de Iris
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Iris.csv")
val spar = SparkSession.builder().getOrCreate()

//Se ejecuta la funcion de RDD para convertir los parametros en valores numericos a la tabla
object RDD{
  def main(args: Array[String]) {
   var parts = 4;
   if( parts == 1.0 ){
  println("Iris-setosa");}
   else if (parts == 2.0){
  println("Iris-versicolor");}
   else if (parts == 3.0){
  println("Iris-virginica");

//Se le declara el vector especifico a cada parte
val data2 = df.map(parts =>
  (parts.label, scaler2.transform(
      Vectors.dense(parts(0).
            toDouble,parts(1).
            toDouble,parts(2).
            toDouble,parts(3).
            toDouble).asML).cache()
            println(data2.first())

//Declaramos el rango de vector del cual vamos a explorar
//val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val testData = splits(1)

//Se especifican los nodos de las capas
// La capa de entrada será 4 (4 características)
val layers = Array[Int](2, 4, 4, 4)

//El 60% de entrenamiento y el 40% de prueba con una semilla 1234L que hace referencia a los pesos (long)
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Modelo de entrenamiento de los nodos
val model = trainer.fit(train)

//Se calcula el modelo transformado
val result = model.transform(test)

//Seleccionamos la etiqueta junto con su predicción de cada nodo
val predictionAndLabels = result.select("prediction", "label")

//A nuestro evaluador se le asignará la métrica de precisión

val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Se imprime el resultado de la precisión de los datos de prueba
val accuracy = evaluator.evaluate(predictions)
println("El error es = " + (1.0 - accuracy))
