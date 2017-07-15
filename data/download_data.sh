
while getopts “size” OPTION
echo "$OPTION"
do
	case $OPTION in
		s)
			echo "Installing SMALL data set"
			echo "Installing 2 GB of data..."
			# SMALL
			mkdir primary_small
			cd primary_small
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_small_v3.zip
			cd ..
			;;
		m)
			echo "Installing MEDIUM data set"
			echo "Installing 10 GB of data..."
			# MEDIUM
			mkdir primary_medium
			cd primary_medium
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_1.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_2.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_3.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_4.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_5.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_files/public_list_primary_v3_medium_21june_2017.csv
			cd ..
			;;
		l)
			echo "Installing LARGE data set (this will take ~30 mins)"
			echo "Installing 50 GB of data..."
			# LARGE
			mkdir primary_full
			cd primary_full
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_files/public_list_primary_v3_full_21june_2017.csv
			cat public_list_primary_v3_full_21june_2017.csv | grep -v UUID | cut -d ',' -f 1 | awk '{printf("https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3/%s.dat\n",$1)}' > full-urls
			cat full-urls | xargs -n 1 -P 100 curl -O
			cd ..
			;;
		a)
			echo "Installing ALL data sets"
			echo "Installing 62 GB of data..."
			echo "Installing small..."
			# SMALL
			mkdir primary_small
			cd primary_small
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_small_v3.zip
			cd ..
			# MEDIUM
			echo "Installing medium..."
			mkdir primary_medium
			cd primary_medium
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_1.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_2.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_3.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_4.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_medium_v3_5.zip
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_files/public_list_primary_v3_medium_21june_2017.csv
			cd ..
			# LARGE
			echo "Installing large... (this will take ~30 mins)"
			mkdir primary_full
			cd primary_full
			curl -O https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_files/public_list_primary_v3_full_21june_2017.csv
			cat public_list_primary_v3_full_21june_2017.csv | grep -v UUID | cut -d ',' -f 1 | awk '{printf("https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3/%s.dat\n",$1)}' > full-urls
			cat full-urls | xargs -n 1 -P 100 curl -O
			cd ..
			;;
		?)
			echo "Please invoke with flag -size with values {s|m|l|a}. Any or all maybe specified."
			echo "Example invokation: sh download_data -size s"
			exit
			;;
	esac
done
echo "Unzipping..."
unzip *.zip
