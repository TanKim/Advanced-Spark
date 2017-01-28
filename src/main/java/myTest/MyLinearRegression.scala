package myTest

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by tkim0 on 2017-01-27.
  */
object MyLinearRegression {

  def run(): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Linear Regression").setMaster("local[*]"))
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()

    val myLinearRegression = new MyLinearRegression(spark)
    myLinearRegression.model()

    spark.stop()
  }

  class MyLinearRegression(spark: SparkSession) {
    def kFoldCrossValidation(): Unit = {

    }

    def model(): Unit = {

      // Libsvm format is consist of the feature vectors and labels like below
      // {label 1:feature1 2:feature2 .. n:featureN}
      val raw = spark.read.format("libsvm").load("C:/dev/data/Linear-Regression/sample_linear_regression_data.txt")
      val splits = raw.randomSplit(Array(0.8, 0.2))
      val train = splits(0).cache()
      val test = splits(1).cache()

      // Create a LinearRegression instance. This instance is an Estimator.
      val lr = new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)

      // Print out the parameters, documentation, and any default values.
      println("LinearRegression parameters:\n" + lr.explainParams() + "\n")

      // Produce Transformer(Model) by using fit method
      // Learn a prediction model using the feature vectors and labels
      val lrModel = lr.fit(train)

      // Now we can optionally save the fitted pipeline to disk
      lrModel.write.overwrite().save("C:/dev/data/model/spark-linear-regression-model")

      // And load it back in during production
      val sameModel = LinearRegressionModel.load("C:/dev/data/model/spark-linear-regression-model")

      // Since lrModel is a Model (i.e., a Transformer produced by an Estimator),
      // we can view the parameters it used during fit().
      // This prints the parameter (name: value) pairs, where names are unique IDs for this
      // LinearRegression instance.
      println("lrModel was fit using parameters: " + lrModel.parent.extractParamMap)

      // Print Coefficients and intercept about Model
      println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

      // Summarize the model over the training set and print out some metrics
      val trainingSummary = lrModel.summary
      println(s"numIterations: ${trainingSummary.totalIterations}")
      println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
      trainingSummary.residuals.show()

      println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      println(s"r2: ${trainingSummary.r2}")

      // prediction only using the feature vectors in test
      val predictions = lrModel.transform(test)
      predictions.select("prediction", "label", "features").show(5)

      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("rmse")
      val rmse = evaluator.evaluate(predictions)
      println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    }
  }
}
