-- MySQL dump 10.13  Distrib 8.4.5, for macos15.2 (arm64)
--
-- Host: localhost    Database: dfjsx
-- ------------------------------------------------------
-- Server version	8.4.5

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `datasets`
--

DROP TABLE IF EXISTS `datasets`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `datasets` (
  `id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(255) DEFAULT NULL,
  `description` text,
  `filename` varchar(255) DEFAULT NULL,
  `raw_data` json DEFAULT NULL,
  `cleaned_data` json DEFAULT NULL,
  `categorical_mappings` json DEFAULT NULL,
  `normalization_params` json DEFAULT NULL,
  `column_renames` json DEFAULT NULL,
  `uploaded_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `s3_key` varchar(512) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `datasets`
--

LOCK TABLES `datasets` WRITE;
/*!40000 ALTER TABLE `datasets` DISABLE KEYS */;
INSERT INTO `datasets` VALUES (7,'rating','rating dataset','rating.csv','[{\"rating\": -1, \"user_id\": 1, \"anime_id\": 20}, {\"rating\": -1, \"user_id\": 1, \"anime_id\": 24}, {\"rating\": -1, \"user_id\": 1, \"anime_id\": 79}, {\"rating\": -1, \"user_id\": 1, \"anime_id\": 226}, {\"rating\": -1, \"user_id\": 1, \"anime_id\": 241}]',NULL,NULL,NULL,NULL,'2025-04-23 15:05:10',NULL),(8,'iris','iris dataset','Iris.csv','[{\"species\": \"setosa\", \"petal_width\": 0.2, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"sepal_length\": 5.1}, {\"species\": \"setosa\", \"petal_width\": 0.2, \"sepal_width\": 3, \"petal_length\": 1.4, \"sepal_length\": 4.9}, {\"species\": \"setosa\", \"petal_width\": 0.2, \"sepal_width\": 3.2, \"petal_length\": 1.3, \"sepal_length\": 4.7}, {\"species\": \"setosa\", \"petal_width\": 0.2, \"sepal_width\": 3.1, \"petal_length\": 1.5, \"sepal_length\": 4.6}, {\"species\": \"setosa\", \"petal_width\": 0.2, \"sepal_width\": 3.6, \"petal_length\": 1.4, \"sepal_length\": 5}]',NULL,NULL,NULL,NULL,'2025-04-23 15:07:02',NULL),(9,'movies','movies dataset','movies.csv','[{\"title\": \"Toy Story (1995)\", \"genres\": \"Adventure|Animation|Children|Comedy|Fantasy\", \"movieId\": 1}, {\"title\": \"Jumanji (1995)\", \"genres\": \"Adventure|Children|Fantasy\", \"movieId\": 2}, {\"title\": \"Grumpier Old Men (1995)\", \"genres\": \"Comedy|Romance\", \"movieId\": 3}, {\"title\": \"Waiting to Exhale (1995)\", \"genres\": \"Comedy|Drama|Romance\", \"movieId\": 4}, {\"title\": \"Father of the Bride Part II (1995)\", \"genres\": \"Comedy\", \"movieId\": 5}]',NULL,NULL,NULL,NULL,'2025-04-25 14:25:48',NULL),(10,'heart','heart dataset','heart.csv','[{\"ca\": 2, \"cp\": 0, \"age\": 52, \"fbs\": 0, \"sex\": 1, \"chol\": 212, \"thal\": 3, \"exang\": 0, \"slope\": 2, \"target\": 0, \"oldpeak\": 1, \"restecg\": 1, \"thalach\": 168, \"trestbps\": 125}, {\"ca\": 0, \"cp\": 0, \"age\": 53, \"fbs\": 1, \"sex\": 1, \"chol\": 203, \"thal\": 3, \"exang\": 1, \"slope\": 0, \"target\": 0, \"oldpeak\": 3.1, \"restecg\": 0, \"thalach\": 155, \"trestbps\": 140}, {\"ca\": 0, \"cp\": 0, \"age\": 70, \"fbs\": 0, \"sex\": 1, \"chol\": 174, \"thal\": 3, \"exang\": 1, \"slope\": 0, \"target\": 0, \"oldpeak\": 2.6, \"restecg\": 1, \"thalach\": 125, \"trestbps\": 145}, {\"ca\": 1, \"cp\": 0, \"age\": 61, \"fbs\": 0, \"sex\": 1, \"chol\": 203, \"thal\": 3, \"exang\": 0, \"slope\": 2, \"target\": 0, \"oldpeak\": 0, \"restecg\": 1, \"thalach\": 161, \"trestbps\": 148}, {\"ca\": 3, \"cp\": 0, \"age\": 62, \"fbs\": 1, \"sex\": 0, \"chol\": 294, \"thal\": 2, \"exang\": 0, \"slope\": 1, \"target\": 0, \"oldpeak\": 1.9, \"restecg\": 1, \"thalach\": 106, \"trestbps\": 138}]',NULL,NULL,NULL,NULL,'2025-04-25 20:50:48','uploads/20250425T205041_heart.csv'),(11,'tv marketing','tv marketing dataset','tvmarketing.csv','[{\"TV\": 230.1, \"Sales\": 22.1}, {\"TV\": 44.5, \"Sales\": 10.4}, {\"TV\": 17.2, \"Sales\": 9.3}, {\"TV\": 151.5, \"Sales\": 18.5}, {\"TV\": 180.8, \"Sales\": 12.9}]',NULL,NULL,NULL,NULL,'2025-04-26 19:47:54','uploads/20250426T194743_tvmarketing.csv');
/*!40000 ALTER TABLE `datasets` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `email` varchar(255) NOT NULL,
  `hashed_password` varchar(255) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `is_verified` tinyint(1) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ix_users_email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (1,'user@example.com','$2b$12$xU86T95KVUGgxCoTOPJRWe.h/e8WOy07eLwTkVza2OoLzPmKO4r6C',1,0,0);
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-26 17:57:36
