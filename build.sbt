name := "courses"

version := "0.1"

scalaVersion := "2.12.12"

val sparkVersion = "3.0.1"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql"  % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

libraryDependencies += "org.json4s" %% "json4s-native" % "3.6.10"