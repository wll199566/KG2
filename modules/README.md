# Issues for using Open IE

As mentioned in the paper, we used [Open IE](https://github.com/dair-iitd/OpenIE-standalone) to extract the triples for each sentence in each hypothesis and support. However, we used Open IE 5.0, instead of Open IE 4.0 (which was used in the paper), due to several improvements. For command of running Open IE, please add `--split` and `--ignore-error` as stated in [README.md](https://github.com/dair-iitd/OpenIE-standalone/blob/master/README.md). Also note that when we run Open IE, it may crash in the middle, so please check out carefully whether it finishes extracting for all hypothesis and supports! If it does not finish, then let it extract for the remaining ones, and pasted the results to the original one.

Open IE command:

`java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar --split --ignore-errors {path_to_split_file} {path_to_output_file}`

