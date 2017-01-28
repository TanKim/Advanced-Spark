import chapter3.{RunRecommender_1st, RunRecommender_2nd}
import chapter4.chapter4.RunRDF_1st
import chapter5.RunKMeans
import myTest.MyLinearRegression
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by tkim0 on 2017-01-14.
  */

object Application {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/winutil/")
    MyLinearRegression.run()
    //RunRecommender_1st.run()
    //RunRecommender_2nd.run()
    //RunRDF_1st.run()
    //RunKMeans.run()
  }
}
