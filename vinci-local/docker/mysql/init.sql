CREATE DATABASE  IF NOT EXISTS `vinci` /*!40100 DEFAULT CHARACTER SET utf8mb4 */;
USE `vinci`;
-- MySQL dump 10.13  Distrib 5.7.33, for Linux (x86_64)
--
-- Host: rm-uf60p2114565x449u4o.mysql.rds.aliyuncs.com    Database: vinci
-- ------------------------------------------------------
-- Server version	5.7.42-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
SET @MYSQLDUMP_TEMP_LOG_BIN = @@SESSION.SQL_LOG_BIN;
SET @@SESSION.SQL_LOG_BIN= 0;

--
-- GTID state at the beginning of the backup
--

SET @@GLOBAL.GTID_PURGED='11d9f446-5278-11ef-9ae1-00163e3a02d1:1-430758,
514a3f81-2c5c-11ee-bd18-00163e285d0f:1-214379,
5391b628-46be-11ef-83f8-00163e36f15d:1-87510,
607a026e-481d-11ee-9243-00163e0a5235:1-1508997,
69a206a0-fdaf-11ee-a3dc-00163e3a0e88:1-473717,
c28b70f8-3aef-11ef-8124-00163e108a1c:1-87559';

--
-- Table structure for table `session_info`
--

DROP TABLE IF EXISTS `session_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `session_info` (
  `session_id` varchar(50) NOT NULL COMMENT '会话id',
  `stream_group_name` varchar(50) NOT NULL DEFAULT '' COMMENT '推送来的视频流的组名',
  `stream_name` varchar(50) NOT NULL DEFAULT '' COMMENT '推送来的视频流名字',
  `is_delete` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user_chat_record`
--

DROP TABLE IF EXISTS `user_chat_record`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `user_chat_record` (
  `id` bigint(20) NOT NULL,
  `session_id` varchar(50) NOT NULL DEFAULT '' COMMENT '会话id',
  `request_text` longtext COMMENT '请求文字',
  `response_text` longtext COMMENT '返回文字',
  `duration` bigint(20) NOT NULL DEFAULT '0' COMMENT '本次问答的持续时间，单位毫秒',
  `request_frame_urls` longtext COMMENT '请求帧图片',
  `response_audio_url` varchar(500) NOT NULL DEFAULT '' COMMENT '返回音频url',
  `response_video_url` varchar(500) NOT NULL DEFAULT '' COMMENT '返回视频url',
  `request_history` longtext COMMENT '本次问答产生的请求历史',
  `response_history` longtext COMMENT '本次问答产生的返回历史',
  `silent` tinyint(1) NOT NULL DEFAULT '0' COMMENT '本次问答是否是silent',
  `status` tinyint(1) NOT NULL DEFAULT '0' COMMENT '本次对话的状态，0正常，1用户主动取消',
  `is_delete` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `retrieval_video_urls` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;
SET @@SESSION.SQL_LOG_BIN = @MYSQLDUMP_TEMP_LOG_BIN;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-10-16 15:47:03