from Data.preprocess_func import categorical_y

LAMBDA_E = 0.0001

class Trainer:
    def __init__(self, ae, latent, classifier):
        self.ae = ae
        self.latent = latent
        self.classifier = classifier
        self.latent_train_loss = 0
        self.ae_train_loss = 0
        self.classifier_train_loss = 0
        self.lambda_e = 0
        self.iterations = 0

    def init_parameters(self):
        self.latent_train_loss = 0
        self.ae_train_loss = 0
        self.classifier_train_loss = 0

    def latent_step(self, X, y, criterion, optim):
        y_cat = categorical_y(y)
        self.ae.eval()
        self.classifier.eval()
        self.latent.train()
        optim.zero_grad()
        z = self.ae.encoder(X)
        y_pred = self.latent(z)
        loss = criterion(y_pred, y_cat)
        self.latent_train_loss += loss.item()
        loss.backward()
        optim.step()

    def ae_step(self, X, y, criterion_ae, criterion_latent, optim):
        self.latent.eval()
        self.classifier.eval()
        self.ae.train()
        optim.zero_grad()
        z, D = self.ae(X, y)
        y_pred = self.latent(z)
        y_b = 1 - y
        y_b_cat = categorical_y(y_b)
        loss = criterion_ae(D, X) + self.lambda_e * criterion_latent(y_pred, y_b_cat)
        self.ae_train_loss += loss.item()
        loss.backward()
        optim.step()
        self.lambda_e += LAMBDA_E
        self.iterations += 1

    def classifier_step(self, X, y, criterion, optim):
        y_cat = categorical_y(y)
        self.latent.eval()
        self.ae.eval()
        self.classifier.train()
        optim.zero_grad()
        y_pred = self.classifier(X)
        loss = criterion(y_pred, y_cat)
        self.classifier_train_loss += loss.item()
        loss.backward()
        optim.step()


class Evaluator:
    def __init__(self, ae, latent, classifier):
        self.ae = ae
        self.latent = latent
        self.classifier = classifier
        self.latent_evaluate_loss = 0
        self.ae_evaluate_loss = 0
        self.classifier_evaluate_loss = 0
        self.latent_evaluate_accuracy = 0
        self.classifier_evaluate_accuracy = 0

    def init_parameters(self):
        self.latent_evaluate_loss = 0
        self.ae_evaluate_loss = 0
        self.classifier_evaluate_loss = 0
        self.latent_evaluate_accuracy = 0
        self.classifier_evaluate_accuracy = 0

    def latent_step(self, X, y, criterion):
        y_cat = categorical_y(y)
        self.ae.eval()
        self.classifier.eval()
        self.latent.eval()
        z = self.ae.encoder(X)
        y_pred = self.latent(z)
        # loss
        loss = criterion(y_pred, y_cat)
        self.latent_evaluate_loss += loss.item()
        # Accuracy
        pred = y_pred.argmax(dim=2, keepdim=True)
        self.latent_evaluate_accuracy += pred.eq(y.view_as(pred)).sum().item()

    def ae_step(self, X, y, criterion):
        self.latent.eval()
        self.classifier.eval()
        self.ae.eval()
        z, D = self.ae(X, y)
        # loss
        loss = criterion(D, X)
        self.ae_evaluate_loss += loss.item()

    def classifier_step(self, X, y, criterion):
        y_cat = categorical_y(y)
        self.ae.eval()
        self.latent.eval()
        self.classifier.eval()
        y_pred = self.classifier(X)
        # loss
        loss = criterion(y_pred, y_cat)
        self.classifier_evaluate_loss += loss.item()
        # Accuracy
        pred = y_pred.argmax(dim=2, keepdim=True)
        self.classifier_evaluate_accuracy += pred.eq(y.view_as(pred)).sum().item()






