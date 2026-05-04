from .common import *

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size, action_size=3):
        try:
            tf_module, keras_layers, keras_models = _load_tensorflow_modules()
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to use DQNAgent. Install it with "
                "`pip install tensorflow` in your project environment."
            ) from exc

        tf_module.get_logger().setLevel('ERROR')
        self.tf = tf_module
        self.layers = keras_layers
        self.models = keras_models
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build neural network"""
        model = self.models.Sequential([
            self.layers.Input(shape=(self.state_size,)),
            self.layers.Dense(64, activation='relu'),
            self.layers.Dropout(0.2),
            self.layers.Dense(32, activation='relu'),
            self.layers.Dropout(0.2),
            self.layers.Dense(16, activation='relu'),
            self.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def update_target_model(self):
        """Update target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.predict_q_values(state)
        return int(np.argmax(act_values))

    def predict_q_values(self, state):
        """Return the raw Q-values for a single state."""
        state_reshaped = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state_reshaped, verbose=0)
        return act_values[0]
    
    def replay(self, batch_size=32):
        """Train with batched experience replay for faster learning."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        actions = np.array([transition[1] for transition in minibatch], dtype=np.int32)
        rewards = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        dones = np.array([transition[4] for transition in minibatch], dtype=np.float32)

        next_q_values = self.target_model.predict(next_states, verbose=0)
        targets = rewards + (1 - dones) * self.gamma * np.max(next_q_values, axis=1)

        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(batch_size), actions] = targets

        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save model"""
        self.model.save_weights(filename)
    
    def load(self, filename):
        """Load model"""
        self.model.load_weights(filename)

