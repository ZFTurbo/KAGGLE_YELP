<?php

set_time_limit(36000);
ini_set('memory_limit', '2596M');
date_default_timezone_set('Europe/London');

$feature_number = 20;

echo "Reading test_index_bn.json\n";
$file = file_get_contents('../mxnet/data/test_index_bn.json');
$fotos_param = json_decode($file, true);

echo "Reading test_photo_to_biz.csv\n";
$file = file_get_contents('../initial_data/test_photo_to_biz.csv');
$lines = explode("\n", $file);
$business = array();
for ($i = 1; $i < count($lines); $i++) {
	$line = trim($lines[$i]);
	if (trim($line) == '')
		continue;
	$arr1 = explode(",", $line);
	
	$photo_id = trim($arr1[0]);
	$business_id = trim($arr1[1]);
	$business[$business_id][] = $photo_id;
}

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
	$f = fopen("test_$feature_number.csv", "w");
	fprintf($f, "Id");
	$total = 0;
	foreach ($unique_params as $k => $u) {
		fprintf($f, ",PBN_".$feature_number."_".$k);
		$param_position[$k] = $total;
		$total++;
	}
	fprintf($f, ",num_photos");
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
		fprintf($f, "\n");
	}

	fclose($f);
}
?>