import tensorflow as tf

class EMANetwork(tf.keras.Model):
    def __init__(self, student, momentum=0.99, name="ema_network"):
        super().__init__(name=name)
        self.student = student  # (f,g) model
        # Clone weights for teacher
        self.teacher = tf.keras.models.clone_model(student)
        self.teacher.set_weights(student.get_weights())
        self.momentum = momentum

    @tf.function
    def update_teacher(self):
        """EMA update: teacher <- γ * teacher + (1-γ) * student"""
        student_weights = self.student.trainable_variables
        teacher_weights = self.teacher.trainable_variables

        for sw, tw in zip(student_weights, teacher_weights):
            tw.assign(self.momentum * tw + (1.0 - self.momentum) * sw)

    @tf.function
    def call(self, inputs, training=False):
        """Forward pass with teacher (f̄,ḡ) for similarity calc"""
        return self.teacher(inputs, training=False)
