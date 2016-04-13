<?php

set_time_limit(36000);
ini_set('memory_limit', '2024M');
date_default_timezone_set('Europe/London');

$feature_number = 20;

echo "Reading train.csv\n";
$file = file_get_contents("../initial_data/train.csv");
$unique_types = array();
$lines = explode("\n", $file);
for ($i = 1; $i < count($lines); $i++) {
	$line = trim($lines[$i]);
	if (trim($line) == '')
		continue;
	$arr1 = explode(",", $line);
	
	$bid = trim($arr1[0]);
	$type_all = trim($arr1[1]);
	$type = explode(" ", $type_all);
	foreach ($type as $t) {
		$t = trim($t);
		if ($t == '')
			continue;
		$unique_types[$t] = 1;
		$train[$bid][$t] = 1;
	}
}

echo "Reading train_photo_to_biz_ids.csv\n";
$file = file_get_contents('../initial_data/train_photo_to_biz_ids.csv');
$pairs = array();
$lines = explode("\n", $file);
$business = array();
for ($i = 1; $i < count($lines); $i++) {
	$line = trim($lines[$i]);
	if (trim($line) == '')
		continue;
	$arr1 = explode(",", $line);
	
	$photo_id = trim($arr1[0]);
	$business_id = trim($arr1[1]);
	$pairs[$photo_id] = $business_id;
	$business[$business_id][] = $photo_id;
}

echo "Reading train_index_bn.json\n";
$file = file_get_contents('../mxnet/data/train_index_bn.json');
$fotos_param = json_decode($file, true);

for ($feature_number = 1; $feature_number <= 20; $feature_number++) {

	echo "Find set of params\n";
	$unique_params = array();
	foreach ($business as $id => $photoarr) {
		foreach ($photoarr as $ph) {
			$jpg = "$ph.jpg";
			// print_r($fotos_param[$jpg]);
			for ($i = 0; $i < $feature_number; $i++) {
				$ind = $fotos_param[$jpg][$i];
				if (!isset($unique_params[$ind]))
					$unique_params[$ind] = 1;
				else
					$unique_params[$ind] += 1;
			}
		}
	}

	# print(count($unique_params));
	# print_r($unique_params);

	$param_position = array();
	echo "Create CSV ($feature_number)\n";
	$f = fopen("train_$feature_number.csv", "w");
	fprintf($f, "Id");
	$total = 0;
	foreach ($unique_params as $k => $u) {
		fprintf($f, ",PBN_".$feature_number."_".$k);
		$param_position[$k] = $total;
		$total++;
	}
	fprintf($f, ",num_photos");
	for ($i = 0; $i <= 8; $i++) {
		fprintf($f, ",label_$i");
	}
	fprintf($f, "\n");

	foreach ($business as $bid => $photoarr) {
		fprintf($f, "$bid");
		unset($line); 
		foreach ($photoarr as $ph) {
			$jpg = "$ph.jpg";
			// print_r($fotos_param[$jpg]);
			for ($i = 0; $i < $feature_number; $i++) {
				$ind = $fotos_param[$jpg][$i];
				if (!isset($line[$ind]))
					$line[$ind] = 1;
				else
					$line[$ind] += 1;
			}
		}
		foreach ($unique_params as $k => $u) {
			fprintf($f, ',');
			if (isset($line[$k])) {
				fprintf($f, $line[$k]);
			}
		}
		fprintf($f, ",".count($photoarr));
		for ($i = 0; $i <= 8; $i++) {
			if (isset($train[$bid][$i]))
				fprintf($f, ",1");
			else
				fprintf($f, ",0");
		}
		fprintf($f, "\n");
	}

	fclose($f);
}
?>