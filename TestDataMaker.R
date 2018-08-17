setwd("/Users/grantholtes/Development/Logit")

logit <- function(x) {
  l<- 1/(1+exp(-x))
  l
}

x1 = rnorm(1000)
x2 = 2*rnorm(1000)
x3 = rnorm(1000)
xb = 3*x1 - 2*x2 + x3 + rnorm(1000)
xb_ = 3*x1 - 4*x2 + 7*x3
y = round(logit(xb))

df = as.data.frame(y)
df$x1 <- x1
df$x2 <- x2
df$x3 <- x3

write.csv(df, file = "data.csv")

logreg = glm(formula = y ~ ., family = binomial(link = "logit"), data = df)
