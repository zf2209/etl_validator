# Bootstrapping ROL Curve
Reverse engineer clients’ premium pricing models with Bootstrapping Linear ROL Curve. This method uses policy structure and LOB specific features, and estimates ROL recursively for fixed exposure split points with customized regularizers, and the model performance is close to the error lower bound for retail line portfolio. The prospective products include premium predictions for arbitrary policy as if by each insurer, ROL curve comparison for every market sub-category, advanced premium outlier detection, client’s model complexity detection.

## Reverse Engineer ROL Curve
This method can fit a ROL curve for each client/LOB, and present in selected market category.
![Reverse Enginner ROL Curve](https://user-images.githubusercontent.com/55263735/111652986-e5ed7a00-87dd-11eb-85c5-3e20c46e5138.png)

## Pricing Arbitrary Policy
This method can use the fitted ROL curve to predict any arbitrary policy. For example, the predicted premium from different clients given left(attachment+deductible), right(left+limit), industry and size.
![Pricing Arbitrary Policy](https://user-images.githubusercontent.com/55263735/111654820-7c6e6b00-87df-11eb-8fa2-8f8c20d88a38.png)

## Premium Outliers Detection
By using the fitted model to predict training portfolio itself, policies with strong deviation of premium and predicted premium will be labeled as outlier and open to further inspection.
![Premium Outliers Detection](https://user-images.githubusercontent.com/55263735/111655269-e2f38900-87df-11eb-9121-2872ef0c696c.png)
