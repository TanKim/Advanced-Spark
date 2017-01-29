package myTest

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionTrainingSummary}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
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

    final val filePath = "C:/dev/data/Linear-Regression/sample_linear_regression_data.txt"
    final val modelPath = "C:/dev/data/model/spark-linear-regression-model"

    def model(): Unit = {

      // Libsvm format is consist of the feature vectors and labels like below
      // {label 1:feature1 2:feature2 .. n:featureN}
      val raw = spark.read.format("libsvm").load(filePath)
      val Array(train, test) = raw.randomSplit(Array(0.9, 0.1), seed = 12345)

      // Create a LinearRegression instance. This instance is an Estimator.
      val lr = new LinearRegression().setMaxIter(10)

      // Print out the parameters, documentation, and any default values.
      //println("LinearRegression parameters:\n" + lr.explainParams() + "\n")

      // Produce Transformer(Model) by using fit method
      // Learn a prediction model using the feature vectors and labels
      val lrModel = lr.fit(train)

      // Since lrModel is a Model (i.e., a Transformer produced by an Estimator),
      // we can view the parameters it used during fit().
      // This prints the parameter (name: value) pairs, where names are unique IDs for this
      // LinearRegression instance.
      // println("lrModel was fit using parameters: " + lrModel.parent.extractParamMap)

      // Now we can optionally save the fitted pipeline to disk
      saveModel(lrModel, modelPath)

      // And load it back in during production
      val sameModel = loadModel(modelPath)

      printModelSummary(lrModel)

      // prediction only using the feature vectors in test
      val lrPredictions = lrModel.transform(test)
      lrPredictions.select("prediction", "label", "features").show(5)

      // We use a ParamGridBuilder to construct a grid of parameters to search over.
      // With 3 values for lr.elasticNetParam and 2 values for lr.regParam,
      // this grid will have 3 x 2 = 6 parameter settings for Validator to choose from.
      val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.01))
        .addGrid(lr.fitIntercept)
        .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
        .build()

      // Use K-fold CrossValidation
      val cv = crossValidation(2, lr, paramGrid)
      val cvModel = cv.fit(train)
      val cvPredictions = cvModel.transform(test)
      cvPredictions.select("prediction", "label", "features").show(5)
      println("[cv RMSE] " + cvModel.getEvaluator.evaluate(cvPredictions))

      // Use TrainValidationSplit
      val tvs = trainValidationSplit(0.8, lr, paramGrid)
      val tvsModel = tvs.fit(train)
      tvsModel.transform(test).select("prediction", "label", "features").show(5)
      println("[tvs RMSE] " + tvsModel.getEvaluator.evaluate(cvPredictions))
    }

    def printModelSummary(model: LinearRegressionModel): Unit ={
      println("###### print simple Model Summary ######")
      // Print Coefficients and intercept about Model
      println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

      // Summarize the model over the training set and print out some metrics
      val trainingSummary = model.summary
      println(s"numIterations: ${trainingSummary.totalIterations}")
      println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
      trainingSummary.residuals.show()

      println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      println(s"r2: ${trainingSummary.r2}")
    }

    def saveModel(model: LinearRegressionModel, path: String): Unit = {
      println(s"[Save Model] $path")
      model.write.overwrite().save(path)
    }

    def loadModel(path: String): LinearRegressionModel = {
      println(s"[Load Model] $path")
      LinearRegressionModel.load(path)
    }

    def crossValidation(k: Int, estimator: LinearRegression, paramGrid: Array[ParamMap]): CrossValidator = {
      println(s"[k-fold cross validation] k=$k")
      // We now treat the LinearRegression as an Estimator, wrapping it in a CrossValidator instance.
      // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
      // Note that the evaluator here is a RegressionEvaluator and its default metric is RMSE.
      new CrossValidator()
        .setEstimator(estimator)
        .setEvaluator(new RegressionEvaluator())
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(k) // use 3+ in practice
    }

    def trainValidationSplit(trainRatio: Double, estimator: LinearRegression, paramGrid: Array[ParamMap]): TrainValidationSplit = {
      println(s"[train Validation Split] ratio=$trainRatio")
      // In this case the estimator is simply the linear regression.
      // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
      new TrainValidationSplit()
        .setEstimator(estimator)
        .setEvaluator(new RegressionEvaluator)
        .setEstimatorParamMaps(paramGrid)
        // 80% of the data will be used for training and the remaining 20% for validation.
        .setTrainRatio(trainRatio)
    }

  }
}
