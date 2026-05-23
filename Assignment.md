Note: You should follow an iterative experimentation process when tuning hyperparameters. After each training experiment, analyze the model’s performance and use the results to justify the next set of parameter adjustments. Each experiment must be documented clearly, including the parameters used, the performance obtained, and the reasoning behind subsequent changes. Where manual experimentation is not required, parameter optimization techniques such as Grid Search may be used to identify suitable hyperparameter combinations.

## **

Report Structure and Requirements**

The submitted report must be clearly structured, well-organized, and written in a formal academic style. While students may choose their own formatting, the report must include all required sections, and each section must directly address the requirements specified in Tasks 1, 2, and 3 of the assignment. The report must include all requested figures, tables, explanations, and justifications.

### **Introduction**

A brief overview of the problem, the objectives of the assignment, and a short description of the dataset used in the study.

### **Task 1: Data**** Handling and Memory Management **

This section must directly address all requirements of Task 1. It must describe the strategy used to load and process the dataset, including techniques applied to manage memory usage. It must explain any data reduction or transformation methods used and justify why these choices were appropriate. It must include a discussion of challenges encountered when working with the dataset and how they were addressed. It must also provide a description of the hardware and software setup used. Evidence of memory usage before and after optimization must be clearly presented.

### **Task 2: ****Exploratory Data Analysis**

This section must include all analyses required in Task 2. It must present the probability density function of traffic across the 10,000 areas and include a discussion of the observed distribution. It must include time series plots for the three specified areas and provide interpretations of their temporal behavior. It must present and interpret the results of stationarity analysis, including the Augmented Dickey-Fuller test. It must include time series decomposition and discuss trend and seasonal patterns. It must include autocorrelation and partial autocorrelation analysis, spatial analysis (such as a heatmap), and a discussion of anomalies or unusual patterns. All figures must be interpreted and linked to meaningful insights about the data.

### **Task 3: Model Design and Implementation**

This section must describe all implemented models as required in Task 3, how they work, and any relevant formulas that demonstrate how they work. It must explain the input representation used for each model, including sequence length and structure, and describe all preprocessing and normalization steps. It must clearly present the structure of each model and the training procedure used.

### **Task 3 ****Results **

This section must present all required outputs from Task 3. It must include prediction plots for each model and each of the three geographical areas, resulting in a total of nine plots. It must include three tables reporting the performance of all models for each area, using MAE, MAPE, and RMSE. It will be great to also provide some information about these evaluation metrics, like how they work and what they really mean, and include their formulas. It must also include a table reporting the training and execution time of each model, together with the hardware details and a description of how the measurements were obtained.

### **Task 3 ****Discussion and Comparative Analysis **

This section must provide a comparative analysis of all models, discussing differences in predictive performance, training time, and suitability for the dataset. It must clearly justify the selection of the best-performing model using both quantitative results and insights derived from the data analysis in Task 2. It must also include reflections on the design and performance of the models and possible improvements.

### **Task 3 ****Failure Analysis **

This section must identify at least one time period where the models perform poorly and provide a clear explanation of the possible causes based on the observed data patterns.

### **Conclusion**

A concise summary of the key findings and insights gained from completing the assignment.

All visualizations must be fully integrated into the report and must not be submitted separately. Each figure must be clearly labeled, include a descriptive caption, and contain readable axis labels, titles, and legends where applicable. Figures must be referenced and discussed within the main text, and each visualization must be accompanied by a clear interpretation explaining what is observed and why it is relevant.

The report must include clear explanations, justifications with solutions previously proposed in the literature, and interpretations throughout. Simply presenting results without reasoning is not sufficient. All sources used, including libraries, tools, online materials, and any external references, must be properly cited using IEEE citation style.

### **References **

References in the IEEE citation style and links to your GitHub repo and demo video should be added here in the references.

Note: Reports that are incomplete, poorly structured, or lacking in clear justification and interpretation may receive reduced marks.
