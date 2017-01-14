import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.log4j.{Level, Logger}

/**
  * Created by tkim0 on 2017-01-14.
  */

object Application {
  def main(args: Array[String]): Unit = {
    val appName = "advanced-spark-application"
    val conf = new SparkConf().setAppName(appName).setMaster("local[*]")
    val sc = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()

    def getCurrentDirectory = new java.io.File(".").getCanonicalPath
    val dirName = getCurrentDirectory
    println(dirName)
    val base = "C:/dev/data/profiledata_06-May-2005/"

    val rawUserArtistData = spark.read.textFile(base + "user_artist_data.txt")
    val rawArtistData = spark.read.textFile(base + "artist_data.txt")
    val rawArtistAlias = spark.read.textFile(base + "artist_alias.txt")

    val runRecommender = new RunRecommender(spark)
    runRecommender.preparation(rawUserArtistData, rawArtistData, rawArtistAlias)
    runRecommender.model(rawUserArtistData, rawArtistData, rawArtistAlias)
    runRecommender.evaluate(rawUserArtistData, rawArtistAlias)
    runRecommender.recommend(rawUserArtistData, rawArtistData, rawArtistAlias)
  }
}
