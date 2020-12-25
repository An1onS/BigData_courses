
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.sql.expressions.Window.partitionBy
import org.apache.spark.sql.functions.{broadcast, col, collect_list, not, row_number, udf}

import scala.collection.mutable
import org.json4s.Extraction.decompose
import org.json4s.native.{prettyJson, renderJValue}

import java.io.{File, FileWriter}

object Main extends App {
  val spark = SparkSession
    .builder()
    .master("local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  //Вариант 31
  val targetCourses = Seq(4040, 35, 786, 831, 1463, 2031)
  val inputPath = "src/main/resources/DO_record_per_line.json"
  val outputPath = "output.json"

  val coursesDs = spark.read
    //.format("json")
    .json(inputPath)
    .as[Course]

  val tokenizer = new RegexTokenizer()
    .setInputCol("desc")
    .setOutputCol("words")
    .setPattern("[^\\w\\dа-яА-Я_Ёё]")
  val wordsData = tokenizer.transform(coursesDs)

  //“danish”, “dutch”, “english”, “finnish”, “french”, “german”, “hungarian”, “italian”, “norwegian”, “portuguese”, “russian”, “spanish”, “swedish” “turkish”
  val stopWords = //эхх
    StopWordsRemover.loadDefaultStopWords("danish")     ++
    StopWordsRemover.loadDefaultStopWords("dutch")      ++
    StopWordsRemover.loadDefaultStopWords("english")    ++
    StopWordsRemover.loadDefaultStopWords("finnish")    ++
    StopWordsRemover.loadDefaultStopWords("french")     ++
    StopWordsRemover.loadDefaultStopWords("german")     ++
    StopWordsRemover.loadDefaultStopWords("hungarian")  ++
    StopWordsRemover.loadDefaultStopWords("italian")    ++
    StopWordsRemover.loadDefaultStopWords("norwegian")  ++
    StopWordsRemover.loadDefaultStopWords("portuguese") ++
    StopWordsRemover.loadDefaultStopWords("russian")    ++
    StopWordsRemover.loadDefaultStopWords("spanish")    ++
    StopWordsRemover.loadDefaultStopWords("swedish")    ++
    StopWordsRemover.loadDefaultStopWords("turkish")

  val remover = new StopWordsRemover()
    .setStopWords(stopWords)
    .setInputCol("words")
    .setOutputCol("filtered")

  val filtered = remover.transform(wordsData)

  val hashingTF = new HashingTF()
    .setInputCol("filtered")
    .setOutputCol("rawFeatures")
    .setNumFeatures(10000)

  val featurizedData = hashingTF.transform(filtered)

  val idf = new IDF()
    .setInputCol("rawFeatures")
    .setOutputCol("features")

  val idfModel = idf.fit(featurizedData)

  val rescaledData = idfModel.transform(featurizedData)

  // дубли
  //rescaledData.filter(col("id").isin(819,20307)).select("id", "name", "desc", "cat", "provider").show(truncate = false)

  val target = rescaledData.filter(col("id").isInCollection(targetCourses))
    .select(
      col("id").as("target_id"),
      col("features").as("target_features"),
      col("lang").as("target_language")
    )

  def cosineSimilarity = udf{ (x : Vector, y : Vector) =>
    val a = x.toArray
    val b = y.toArray
    val l1 = scala.math.sqrt(a.map(x => x * x).sum)
    val l2 = scala.math.sqrt(b.map(y => y * y).sum)
    val scalar = a.zip(b).map(p => p._1 * p._2).sum
    scalar / (l1 * l2)
  }

  val courses = rescaledData.filter(not(
    col("id").isInCollection(targetCourses)
  )).join(
    broadcast(target),
    col("lang") === col("target_language")
  ).withColumn("similarity",
    cosineSimilarity(col("features"),
                     col("target_features")))

  val result = courses
    .withColumn("rank", row_number().over(
      partitionBy(col("target_id"))
        .orderBy(
          col("similarity").desc_nulls_last,
          col("name")
        )
    )).filter(col("rank") < 11)
    .cache()

  val output = result.select("target_id", "id")
    .orderBy("target_id", "rank")
    .groupBy("target_id")
    .agg(collect_list("id").as("ids"))
    .withColumn("maps", functions.map('target_id, 'ids))
    .cache()

  val outputMaps = output.select("maps").as[Map[String,Array[BigInt]]].collect()

  val outputMap = mutable.Map.empty[String,Array[BigInt]]

  outputMaps.map(m => outputMap += m.head)

  implicit val formats = org.json4s.DefaultFormats

  val jsonString = prettyJson(renderJValue(decompose(outputMap.toMap)))

  val brackets = """(\[)""".r.replaceAllIn(jsonString,"$1\n   ")
  val text = """(,)""".r.replaceAllIn(brackets,"$1\n   ")
  println(text)

  val fileWriter = new FileWriter(new File(outputPath))
  fileWriter.write(text)
  fileWriter.close()

}
