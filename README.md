# Relation Extraction using TensorFlow
Relation Extraction research done for [the Advanced Natural Language Processing](https://cs.nyu.edu/courses/fall17/CSCI-GA.2591-001/) course at NYU. It was built to enhance the [JetLite](https://cs.nyu.edu/courses/fall17/CSCI-GA.2591-001/JetLite.html) system by Prof. Ralph Grishman.

This repository only contains the files that I personally worked on, as a distilled version of my contributions. It consists of the Python scripts that generate the model, and the Java classes that load and run the model. The whole thing can be found [here](https://github.com/dorukkilitcioglu/jetLite/tree/tensorflow). The presentation of the work I've done can be found [here](https://drive.google.com/open?id=1kIAs3XkIS0_SyRj2qUhTgLcbFDghASwyNaTkM3g9te8&lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3B4ZjFMwDfRCaQd7T8uk6CIQ%3D%3D).

## Relation Extraction
There are 3 different ways of classification schemes (or relation type details) for relation extraction that are used here. First is the basic relation types, where only the 6 main ACE relation types are used (7 total with `other` class representing no relation). Then there are the subtypes, where two relations only match if their subtypes match. There are 19 total classes in this scheme (18 + `other`). The last is the subclass with ordering, where two relations only match iff their subtypes match and the relationship order is preserved. There are 37 total classes in this scheme (36 + `other`).

### Choosing a type detail
For the 3 different levels of details in classification, you need to set the `relation_detail` variable, with the values `basic`, `subtype`, and `subtype_with_order` for 7, 19, 37 classes respectively.

## Running the code
Detailed instructions are found in the [original repo](https://github.com/dorukkilitcioglu/jetLite/tree/tensorflow). If you can generate your own training data, you can also simply use the Python scripts provided here.

### TensorFlow
Make sure that you have Tensorflow installed. For Java integration, version 1.3.0 is needed. The Python scripts should work with the new releases, until there is a breaking change in the API.

### Necessary Data Files
The project is missing the ACE files, which are available through [LDC](https://catalog.ldc.upenn.edu/LDC2005T09). It is converted to an intermediate representation by my extension of `JetLite`, and is read by `loader.py`. The other requirement is the GloVe pretrained word embeddings file, which can be found on the [Stanford NLP's GloVe website](https://nlp.stanford.edu/projects/glove/).
