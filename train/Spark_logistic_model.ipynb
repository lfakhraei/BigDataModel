{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "2e0c2e55-277a-4b9e-bc8c-0f7f85850f02",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "from pyspark.sql import SparkSession\n",
    "from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler\n",
    "from pyspark.ml.classification import LogisticRegression\n",
    "from pyspark.ml.evaluation import BinaryClassificationEvaluator\n",
    "from pyspark.ml.tuning import CrossValidator, ParamGridBuilder\n",
    "from pyspark.ml import Pipeline\n",
    "from pyspark.sql import functions as F\n",
    "from pyspark.ml.feature import StringIndexer\n",
    "from pyspark.sql.functions import col\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "c1439ff9-632d-4c50-868a-856c0cc6a897",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "# Create a SparkSession\n",
    "spark = SparkSession.builder \\\n",
    "    .appName(\"Income Prediction\") \\\n",
    "    .getOrCreate()\n",
    "\n",
    "# Read the train and test data\n",
    "train = spark.read.csv(\"s3://adhoc-query-data/Leila/check_data/train.tsv\", sep=\"\\t\", header=True, inferSchema=True)\n",
    "test = spark.read.csv(\"s3://adhoc-query-data/Leila/check_data/test.tsv\", sep=\"\\t\", header=True, inferSchema=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "de22b4c3-ba99-4af5-9bf0-ddceb6e6ee32",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+---+---+---------+------+---------+-------------+--------------+-----------------+------------+-----+----+------------+------------+--------------+--------------+\n|id |age|workclass|fnlwgt|education|education-num|marital-status|occupation       |relationship|race |sex |capital-gain|capital-loss|hours-per-week|native-country|\n+---+---+---------+------+---------+-------------+--------------+-----------------+------------+-----+----+------------+------------+--------------+--------------+\n|0  |25 |Private  |226802|11th     |7            |Never-married |Machine-op-inspct|Own-child   |Black|Male|0           |0           |40            |United-States |\n+---+---+---------+------+---------+-------------+--------------+-----------------+------------+-----+----+------------+------------+--------------+--------------+\nonly showing top 1 row\n\n"
     ]
    }
   ],
   "source": [
    "test.show(1,truncate=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "d9743ea3-113b-4966-aa82-755e3402448f",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of missing values in 'id' column is 0\nNumber of missing values in 'age' column is 0\nNumber of missing values in 'workclass' column is 1836\nNumber of missing values in 'fnlwgt' column is 0\nNumber of missing values in 'education' column is 0\nNumber of missing values in 'education-num' column is 0\nNumber of missing values in 'marital-status' column is 0\nNumber of missing values in 'occupation' column is 1843\nNumber of missing values in 'relationship' column is 0\nNumber of missing values in 'race' column is 0\nNumber of missing values in 'sex' column is 0\nNumber of missing values in 'capital-gain' column is 0\nNumber of missing values in 'capital-loss' column is 0\nNumber of missing values in 'hours-per-week' column is 0\nNumber of missing values in 'native-country' column is 583\nNumber of missing values in 'income' column is 0\n"
     ]
    }
   ],
   "source": [
    "for col in train.columns:\n",
    "    num_missing = train.filter(train[col].isNull()).count()\n",
    "    print(f\"Number of missing values in '{col}' column is {num_missing}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "86b39191-5acf-4fdf-9765-329b8868ae58",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "# Preprocess the data\n",
    "train = train.withColumn(\"label\", (train[\"income\"] == \">50K\").cast(\"int\"))\n",
    "train = train.drop(\"income\")\n",
    "columns_to_drop = [\"education-num\", \"fnlwgt\", \"capital-loss\"]\n",
    "train = train.drop(*columns_to_drop)\n",
    "test = test.drop(*columns_to_drop)\n",
    "\n",
    "# Fill missing values in categorical columns with mode\n",
    "categorical_columns = [\"workclass\", \"occupation\", \"native-country\"]\n",
    "for col in categorical_columns:\n",
    "    mode_value = train.select(col).groupBy(col).count().orderBy(\"count\", ascending=False).first()[col]\n",
    "    train = train.na.fill({col: mode_value})\n",
    "    test = test.na.fill({col: mode_value})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "71820f31-0150-4953-a052-f437571a7ff4",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "no_work_or_pay = ['Never-worked', 'Without-pay']\n",
    "train = train.withColumn('workclass', F.when(train['workclass'].isin(no_work_or_pay), 'no_work_or_pay').otherwise(train['workclass']))\n",
    "\n",
    "postgrad_education = ['Masters', 'Prof-school', 'Doctorate']\n",
    "basic_education = ['Preschool', '1st-4th', '5th-6th', '11th', '9th', '7th-8th', '10th', '12th']\n",
    "associate = ['Assoc-acdm', 'Assoc-voc']\n",
    "train = train.withColumn('education', F.when(train['education'].isin(postgrad_education), 'postgrad').when(train['education'].isin(basic_education), 'basic_education').when(train['education'].isin(associate), 'associate').otherwise(train['education']))\n",
    "\n",
    "single = ['Never-married', 'Separated']\n",
    "after_marriage = ['Married-spouse-absent', 'Widowed', 'Divorced']\n",
    "AF_civ = ['Married-AF-spouse', 'Married-civ-spouse']\n",
    "train = train.withColumn('marital-status', F.when(train['marital-status'].isin(single), 'single').when(train['marital-status'].isin(after_marriage), 'after_marriage').when(train['marital-status'].isin(AF_civ), 'AF_civ').otherwise(train['marital-status']))\n",
    "\n",
    "service = ['Priv-house-serv', 'Other-service', 'Handlers-cleaners']\n",
    "other = ['Armed-Forces', 'Farming-fishing', 'Machine-op-inspct','Adm-clerical']\n",
    "move_repair = ['Transport-moving', 'Craft-repair']\n",
    "sale_support = ['Sales', 'Tech-support']\n",
    "train = train.withColumn('occupation', F.when(train['occupation'].isin(service), 'service').when(train['occupation'].isin(other), 'other').when(train['occupation'].isin(move_repair), 'move_repair').when(train['occupation'].isin(sale_support), 'sale_support').otherwise(train['occupation']))\n",
    "\n",
    "other_countries = ['Trinadad&Tobago', 'Cambodia', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland', 'Holand-Netherlands']\n",
    "wealthy_countries = ['Ireland', 'United-States', 'Cuba', 'China', 'Greece', 'Hong', 'Philippines', 'Germany', 'Canada', 'England', 'Italy', 'Japan', 'Taiwan', 'India', 'France', 'Iran']\n",
    "less_wealthy = ['Dominican-Republic', 'Columbia', 'Guatemala', 'Mexico', 'Nicaragua', 'Peru', 'Vietnam', 'El-Salvador', 'Haiti', 'Puerto-Rico', 'Portugal']\n",
    "train = train.withColumn('native-country', F.when(train['native-country'].isin(other_countries), 'other').when(train['native-country'].isin(wealthy_countries), 'wealthy_countries').when(train['native-country'].isin(less_wealthy), 'less_wealthy').otherwise(train['native-country']))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "03b52861-6f42-4526-8636-5ccefcd64160",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Out[6]: [Row(marital-status='single'),\n Row(marital-status='after_marriage'),\n Row(marital-status='AF_civ')]"
     ]
    }
   ],
   "source": [
    "train.select('marital-status').distinct().collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "9ad63443-3a89-4648-be31-4dd412af1b8e",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "# Get the columns with string data type\n",
    "df_cat = train.select([col for col, dtype in train.dtypes if dtype == 'string'])\n",
    "\n",
    "# Get the list of categorical attribute names\n",
    "attr_cat = df_cat.columns\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "f47c333c-00c3-43c3-8533-39542f2b3d23",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "# Convert categorical variables into one-hot encoded format\n",
    "\n",
    "for attr in attr_cat:\n",
    "    indexer = StringIndexer(inputCol=attr, outputCol=attr+\"_index\")\n",
    "    encoder = OneHotEncoder(inputCol=attr+\"_index\", outputCol=attr+\"_encoded\")\n",
    "    pipeline = Pipeline(stages=[indexer, encoder])\n",
    "    train = pipeline.fit(train).transform(train)\n",
    "    test = pipeline.fit(test).transform(test)\n",
    "    # Drop the original categorical column\n",
    "    train = train.drop(attr)\n",
    "    test = test.drop(attr)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "65351023-2fc8-4d23-b12d-366243006a6c",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "X = train.select([col for col in train.columns if col not in ['label', 'id']])\n",
    "\n",
    "# Select only the 'label' column to create target dataframe y\n",
    "y = train.select('label')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "3db3e3b1-b486-480d-89f6-8da84fbe1d4d",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "# Ensure both train and test datasets have the same features as the training dataset\n",
    "features_ls = X.columns\n",
    "test = test.select(*features_ls).fillna(0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "bfdc3206-5ea8-49cf-afe5-c56c27e55202",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC: 0.9030538945304412\nMean AUC score: 0.8881171548231684\n"
     ]
    }
   ],
   "source": [
    "# Get the schema of the DataFrame\n",
    "train_schema = train.schema\n",
    "\n",
    "# Identify string columns in the schema\n",
    "string_columns = [col.name for col in train_schema if col.dataType == \"string\"]\n",
    "\n",
    "# Filter the string columns based on the list of categorical columns\n",
    "categorical_columns = [col for col in string_columns if col in [\"workclass\", \"occupation\", \"native-country\"]]\n",
    "\n",
    "# Define the stages for StringIndexer and OneHotEncoder\n",
    "indexers = [StringIndexer(inputCol=col, outputCol=col+\"_index\", handleInvalid=\"keep\") for col in categorical_columns]\n",
    "encoders = [OneHotEncoder(inputCol=col+\"_index\", outputCol=col+\"_encoded\") for col in categorical_columns]\n",
    "\n",
    "# Assemble all features into a single feature vector\n",
    "assembler = VectorAssembler(inputCols=[col+\"_encoded\" for col in categorical_columns] + [col for col in train.columns if col not in categorical_columns + [\"label\"]],\n",
    "                            outputCol=\"features\")\n",
    "\n",
    "# Apply the VectorAssembler to the train DataFrame\n",
    "train = assembler.transform(train)\n",
    "\n",
    "\n",
    "# Split dataset into train and holdout sets\n",
    "train, holdout = train.randomSplit([0.8, 0.2], seed=42)\n",
    "\n",
    "# Define Logistic Regression model\n",
    "lr = LogisticRegression(featuresCol='features', labelCol='label')\n",
    "\n",
    "# Fit the Logistic Regression model\n",
    "model = lr.fit(train)\n",
    "\n",
    "# Make predictions\n",
    "predictions = model.transform(holdout)\n",
    "\n",
    "# Evaluate the model\n",
    "evaluator = BinaryClassificationEvaluator(rawPredictionCol=\"rawPrediction\", labelCol=\"label\")\n",
    "auc = evaluator.evaluate(predictions, {evaluator.metricName: \"areaUnderROC\"})\n",
    "print(\"ROC AUC:\", auc)\n",
    "\n",
    "\n",
    "\n",
    "# Perform cross-validation\n",
    "\n",
    "\n",
    "paramGrid = ParamGridBuilder() \\\n",
    "    .addGrid(lr.regParam, [0.1, 0.01]) \\\n",
    "    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \\\n",
    "    .build()\n",
    "\n",
    "crossval = CrossValidator(estimator=lr,\n",
    "                          estimatorParamMaps=paramGrid,\n",
    "                          evaluator=BinaryClassificationEvaluator(),\n",
    "                          numFolds=5)\n",
    "\n",
    "cvModel = crossval.fit(train)\n",
    "\n",
    "# Get average AUC across all folds\n",
    "avg_auc = cvModel.avgMetrics[0]\n",
    "print(\"Mean AUC score:\", avg_auc)\n"
   ]
  }
 ],
 "metadata": {
  "application/vnd.databricks.v1+notebook": {
   "dashboards": [],
   "language": "python",
   "notebookMetadata": {
    "pythonIndentUnit": 4
   },
   "notebookName": "Logostic_regression",
   "widgets": {}
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
