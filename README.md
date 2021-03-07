# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Wen Xu

*Answer all questions in the notebook here.  You should also write whatever high-level documentation you feel you need to here.*

Part 1 - Preprocessing

From the given dataset I generated a simple list by removing 't' and '\n'.  I also converted all letters to be lower case, removed unformative words such as '""""' or common stop words in English (a predefined array of stop words is included in a2.py).


Part 2 - Create instances

I created Instance objects as required. In particular i used '$'(or '&') as the padding letter when the context before (or after) NE is less than 5.


Part 3 - Create the table and splitting

I removed all occurances of '$' and '&' as they contribute nothing to the context analysis. Then I created TF-IDF as the features table. Splitting of data set is implemented as required.
Sometimes test_y[0] seems to throw an exception for some unknown reason. But test_y seems to contain valid data and the program would be able to continue to run the following tasks.

Part 4 - Training the model

None

Part 5 - Evaluation

I noticed that the type 'geo' which is the most frequent NE type in the train set has the highest accuracy when validating with test set. In general I can conclude that the types ('gpe', 'tim', 'per', 'org') that occur most in the taining set will lead to higher accuracy in test set.


Bonus B - Expanding the feature space

When adding the context (five terms before and after the NE), I also include the POS of the term as part of the feature. After padding, the size of feature vector becomes doubled. With the expanded features space, the trained model leads to better accuracy when verifying with the test set.