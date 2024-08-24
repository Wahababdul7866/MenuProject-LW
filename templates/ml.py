import streamlit as st

def show_long_text():
    """Displays a long paragraph of text when a button is clicked."""

    if st.button("Show Long Text"):
        st.markdown("""

            **We will be exploring each of these techniques in a detailed manner now.**

1. Next or Previous Value
For time-series data or ordered data, there are specific imputation techniques. These techniques take into consideration the dataset's sorted structure, wherein nearby values are likely more comparable than far-off ones. The next or previous value inside the time series is typically substituted for the missing value as part of a common method for imputed incomplete data in the time series. This strategy is effective for both nominal and numerical values.

2. K Nearest Neighbors
The objective is to find the k nearest examples in the data where the value in the relevant feature is not absent and then substitute the value of the feature that occurs most frequently in the group.

3. Maximum or Minimum Value
You can use the minimum or maximum of the range as the replacement cost for missing values if you are aware that the data must fit within a specific range [minimum, maximum] and if you are aware from the process of data collection that the measurement instrument stops recording and the message saturates further than one of such boundaries. For instance, if a price cap has been reached in a financial exchange and the exchange procedure has indeed been halted, the missing price can be substituted with the exchange boundary's minimum value.

4. Missing Value Prediction
Using a machine learning model to determine the final imputation value for characteristic x based on other features is another popular method for single imputation. The model is trained using the values in the remaining columns, and the rows in feature x without missing values are utilized as the training set. 

Depending on the type of feature, we can employ any regression or classification model in this situation. In resistance training, the algorithm is used to forecast the most likely value of each missing value in all samples.

A basic imputation approach, such as the mean value, is used to temporarily impute all missing values when there is missing data in more than a feature field. Then, one column's values are restored to missing. After training, the model is used to complete the missing variables. In this manner, an is trained for every feature that has a missing value up until a model can impute all of the missing values.

5. Most Frequent Value
The most frequent value in the column is used to replace the missing values in another popular technique that is effective for both nominal and numerical features.

6. Average or Linear Interpolation
The average or linear interpolation, which calculates between the previous and next accessible value and substitutes the missing value, is similar to the previous/next value imputation but only applicable to numerical data. Of course, as with other operations on ordered data, it is crucial to accurately sort the data in advance, for example, in the case of time series data, according to a timestamp.

7. (Rounded) Mean or Moving Average or Median Value
Median, Mean, or rounded mean are further popular imputation techniques for numerical features. The technique, in this instance, replaces the null values with mean, rounded mean, or median values determined for that feature across the whole dataset. It is advised to utilize the median rather than the mean when your dataset has a significant number of outliers.

8. Fixed Value
Fixed value imputation is a universal technique that replaces the null data with a fixed value and is applicable to all data types. You can impute the null values in a survey using "not answered" as an example of using fixed imputation on nominal features.

Since we have explored single imputation, its importance, and its techniques, let us now learn about Multiple imputations.

        """)




def show_long_text1():
    """Displays a long paragraph of text when a button is clicked."""

    if st.button("Show Long Text"):
        st.markdown(""" **What Happens to the Dropped Category's Weight?**
When you drop a category from a categorical variable, you're essentially removing all instances that belong to that category from your dataset. Here's what happens to the "weight":

*Data Removal: The data representing the dropped category is literally removed from your dataset. It's no longer used in any analysis or modeling.

*Shift in Proportions: The relative proportions of the remaining categories shift. If the dropped category was a significant portion of the data, the other categories will have a greater relative weight in the remaining dataset.

**Potential Consequences:**
Dropping a category can have significant consequences, depending on the context:

*Bias: If you drop a category that is important to the outcome you're trying to predict (e.g., a gender category in a study about income), you can introduce bias into your analysis. Your model might not be able to accurately generalize to the full population if a relevant group is missing.

*Reduced Model Performance: Dropping important categories can lead to worse model performance, especially if the dropped category had a significant impact on the outcome variable.

*Misinterpretation of Results: The results of your analysis might not be generalizable to the real world if you've dropped important categories.

**Analyzing the Impact**
Here are ways to analyze the impact of dropping categories:

*Visualize the Data: Before and after dropping a category, create visualizations like bar charts or pie charts to see how the distribution of categories changes.

*Compare Model Performance: If you're using machine learning, train models both with and without the dropped category. Compare their performance metrics (accuracy, precision, recall, etc.) to see if the dropped category was important for the model's predictions.

*Use One-Hot Encoding: If you're using one-hot encoding, consider encoding the dropped category with a value of 0 to track its influence in your analysis, even though it's not directly used in the model.

*Consider Alternatives: Instead of dropping categories, explore alternatives like:

One-Hot Encoding: Create binary variables for each category, even the dropped one.

Ordinal Encoding: Encode the categories with meaningful numerical values if there is an order to them.

Target Encoding: Replace categorical values with the mean of the target variable for each category.
""")



def show_long_text2():
    """about different initializers and their use cases"""

    if st.button("Show Long Text"):
        st.markdown("""
**Here's a breakdown of various initializers, their use cases, and their characteristics:**
1. Zero Initialization
Concept: All weights and biases are initialized to 0.
Use Case: Not recommended for most cases.
Drawbacks: Leads to symmetry in the network, making all neurons learn the same thing. It can cause the network to get stuck in a state where no learning occurs.
2. Constant Initialization
Concept: All weights and biases are initialized to a constant value (other than 0).
Use Case: Rarely used.
Drawbacks: Similar issues to zero initialization; can lead to biases in the network and slow convergence.
3. Random Initialization
Concept: Weights and biases are initialized to random values drawn from a distribution.
Use Case: Most common approach; helps break symmetry and allows neurons to learn differently.
Types:
Uniform Distribution (tf.keras.initializers.RandomUniform): Draws values uniformly from a given range.
Normal Distribution (tf.keras.initializers.RandomNormal): Draws values from a normal distribution with specified mean and standard deviation.
Considerations:
Xavier/Glorot Initialization (tf.keras.initializers.GlorotUniform, tf.keras.initializers.GlorotNormal): Designed to maintain the variance of activations as the signal propagates through the network. Suitable for most networks.
He Initialization (tf.keras.initializers.HeUniform, tf.keras.initializers.HeNormal): Specifically for ReLU activations. Scales the weights based on the number of inputs to the layer.
4. Other Initializers
Orthogonal Initialization (tf.keras.initializers.Orthogonal): Initializes weights with orthogonal matrices to preserve information flow. Useful for recurrent neural networks (RNNs).
Lecun Normal Initialization (tf.keras.initializers.LecunNormal): Similar to He initialization but uses the square root of 2 divided by the number of inputs. Suitable for sigmoid and tanh activations.
Variance Scaling Initialization (tf.keras.initializers.VarianceScaling): Allows customization of scaling based on the activation function, number of inputs, and distribution (normal or uniform).

""")



def show_long_text3():
    """Find the use cases of optimizers"""

    if st.button("Show Long Text"):
        st.markdown("""

**Here's a breakdown of various initializers, their use cases, and their characteristics:**
1. Zero Initialization
Concept: All weights and biases are initialized to 0.
Use Case: Not recommended for most cases.
Drawbacks: Leads to symmetry in the network, making all neurons learn the same thing. It can cause the network to get stuck in a state where no learning occurs.
2. Constant Initialization
Concept: All weights and biases are initialized to a constant value (other than 0).
Use Case: Rarely used.
Drawbacks: Similar issues to zero initialization; can lead to biases in the network and slow convergence.
3. Random Initialization
Concept: Weights and biases are initialized to random values drawn from a distribution.
Use Case: Most common approach; helps break symmetry and allows neurons to learn differently.
Types:
Uniform Distribution (tf.keras.initializers.RandomUniform): Draws values uniformly from a given range.
Normal Distribution (tf.keras.initializers.RandomNormal): Draws values from a normal distribution with specified mean and standard deviation.
Considerations:
Xavier/Glorot Initialization (tf.keras.initializers.GlorotUniform, tf.keras.initializers.GlorotNormal): Designed to maintain the variance of activations as the signal propagates through the network. Suitable for most networks.
He Initialization (tf.keras.initializers.HeUniform, tf.keras.initializers.HeNormal): Specifically for ReLU activations. Scales the weights based on the number of inputs to the layer.
4. Other Initializers
Orthogonal Initialization (tf.keras.initializers.Orthogonal): Initializes weights with orthogonal matrices to preserve information flow. Useful for recurrent neural networks (RNNs).
Lecun Normal Initialization (tf.keras.initializers.LecunNormal): Similar to He initialization but uses the square root of 2 divided by the number of inputs. Suitable for sigmoid and tanh activations.
Variance Scaling Initialization (tf.keras.initializers.VarianceScaling): Allows customization of scaling based on the activation function, number of inputs, and distribution (normal or uniform).
Use Case Guidelines:
ReLU Activations: Use He initialization (HeUniform or HeNormal).
Sigmoid or Tanh Activations: Use Xavier/Glorot initialization (GlorotUniform or GlorotNormal) or Lecun Normal initialization.
Recurrent Neural Networks (RNNs): Consider Orthogonal initialization for the recurrent weights.
Experiment: Try different initializers and see what works best for your specific network architecture and dataset.

""")


def show_long_text4():
    """Find which activation function works with which type of pooling."""

    if st.button("Show Long Text"):
        st.markdown("""
**Activation Functions**

1.ReLU (Rectified Linear Unit):
Advantages: Fast computation, avoids vanishing gradients.
Drawbacks: Can lead to "dying ReLU" problem (neurons can get stuck).
2.Sigmoid:
Advantages: Outputs values between 0 and 1, useful for binary classification.
Drawbacks: Can suffer from vanishing gradients in deep networks.
3.Tanh (Hyperbolic Tangent):
Advantages: Similar to sigmoid but with output range of -1 to 1.
Drawbacks: Can also suffer from vanishing gradients.
4.Softmax:
Advantages: Outputs a probability distribution over multiple classes.
Drawbacks: Used for multi-class classification.
5.Linear:
Advantages: No non-linearity, used in output layers for regression problems.
Drawbacks: Can be less robust for complex tasks.

**Pooling Layers**
1.Max Pooling:
Concept: Takes the maximum value from a region of the input.
Advantages: Preserves the most important features, robust to small shifts in input.
Drawbacks: Can lose some information.
2.Average Pooling:
Concept: Calculates the average value from a region of the input.
Advantages: Less sensitive to outliers than max pooling.
Drawbacks: Can blur features and lose detail.
3.Global Average Pooling (GAP):
Concept: Applies average pooling to the entire feature map.
Advantages: Reduces the number of parameters and can help prevent overfitting.
Drawbacks: Can lose local information.


**Common Pairings and Considerations:**
*ReLU + Max Pooling: A very common combination, often used together for convolutional neural networks (CNNs). Max pooling works well with ReLU because it preserves the strongest activations.
*ReLU + Average Pooling: Can also be used but may not be as effective as max pooling with ReLU.
*Sigmoid/Tanh + Max Pooling/Average Pooling: These combinations are less common because they are more prone to vanishing gradients in deeper networks.
*Softmax + Global Average Pooling: A popular pairing for image classification tasks, especially when using CNNs with a few final layers.

**General Guidelines:**
1.Max Pooling: A good choice with ReLU activations, especially when you want to preserve the most important features.
2.Average Pooling: Consider using average pooling with ReLU if you are concerned about outliers in the input.
3.Global Average Pooling: Use GAP for image classification tasks with CNNs to help prevent overfitting.
4.Experiment: The best combination depends on your specific network architecture, dataset, and task. Try different pairings and see which ones perform best.

**Key Considerations:**
1.Network Architecture: The specific network architecture will influence your choices. CNNs typically use pooling, while feedforward networks often don't.
2.Dataset Characteristics: The complexity and characteristics of your data will also play a role. For example, if your data is noisy, average pooling might be more robust.
3.Task: The type of task (classification, regression, etc.) will influence the choice of activation function.

""")