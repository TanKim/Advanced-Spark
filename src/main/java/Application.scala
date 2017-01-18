import chapter3.{RunRecommender_1st, RunRecommender_2nd}

/**
  * Created by tkim0 on 2017-01-14.
  */

object Application {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/winutil/")
    RunRecommender_1st.run()
    //RunRecommender_2nd.run()
    //RunRDF_1st.run()
  }
}
