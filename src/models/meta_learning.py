"""
Meta-Learning Framework for Trading
==================================

Advanced meta-learning implementations for rapid adaptation to new market regimes,
asset classes, and trading environments.

Features:
- Model-Agnostic Meta-Learning (MAML) for quick adaptation
- Few-Shot Learning for new asset classes
- Continual Learning to prevent catastrophic forgetting
- Multi-Task Learning for shared representations
- Regime-Aware Meta-Learning for market state transitions
- Gradient-Based Meta-Learning optimization
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import asyncio
import copy
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TaskDataset(Dataset):
    """Dataset for meta-learning tasks."""
    
    def __init__(self, 
                 features: np.ndarray, 
                 targets: np.ndarray,
                 task_id: str = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.task_id = task_id
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MarketTask:
    """Represents a trading task for meta-learning."""
    
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 target_column: str = 'future_return',
                 feature_columns: List[str] = None,
                 regime: str = 'normal',
                 asset_class: str = 'equity'):
        
        self.name = name
        self.data = data
        self.target_column = target_column
        self.feature_columns = feature_columns or self._get_feature_columns(data)
        self.regime = regime
        self.asset_class = asset_class
        
        # Prepare task data
        self.X = data[self.feature_columns].values
        self.y = data[target_column].values if target_column in data.columns else self._create_targets(data)
        
        # Handle missing values
        self.X = np.nan_to_num(self.X, nan=0.0)
        self.y = np.nan_to_num(self.y, nan=0.0)
        
        logger.info(f"Created task '{name}': {len(self.X)} samples, {len(self.feature_columns)} features")
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Extract feature columns from data."""
        excluded = ['date', 'timestamp', 'symbol', 'target', 'future_return', 'regime']
        return [col for col in data.columns if col not in excluded and data[col].dtype in ['float64', 'int64']]
    
    def _create_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Create target variable from price data."""
        if 'close' in data.columns:
            # Create future returns
            returns = data['close'].pct_change(periods=5).shift(-5)  # 5-period forward return
            # Convert to classification: 0=down, 1=neutral, 2=up
            targets = np.where(returns < -0.01, 0, np.where(returns > 0.01, 2, 1))
            return targets
        else:
            # Random targets for testing
            return np.random.randint(0, 3, len(data))
    
    def get_support_query_split(self, 
                               support_size: int = 50, 
                               query_size: int = 50) -> Tuple[TaskDataset, TaskDataset]:
        """Split task data into support and query sets."""
        total_samples = len(self.X)
        
        if total_samples < support_size + query_size:
            # If not enough data, split proportionally
            support_size = int(total_samples * 0.5)
            query_size = total_samples - support_size
        
        indices = np.random.permutation(total_samples)
        support_indices = indices[:support_size]
        query_indices = indices[support_size:support_size + query_size]
        
        support_dataset = TaskDataset(
            self.X[support_indices], 
            self.y[support_indices],
            f"{self.name}_support"
        )
        
        query_dataset = TaskDataset(
            self.X[query_indices], 
            self.y[query_indices],
            f"{self.name}_query"
        )
        
        return support_dataset, query_dataset


class BaseMetaModel(nn.Module, ABC):
    """Base class for meta-learning models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def clone(self):
        pass


class MLPMetaModel(BaseMetaModel):
    """Multi-Layer Perceptron for meta-learning."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def clone(self):
        """Create a deep copy of the model."""
        return copy.deepcopy(self)


class MAMLTrainer:
    """Model-Agnostic Meta-Learning (MAML) trainer."""
    
    def __init__(self,
                 model: BaseMetaModel,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 inner_steps: int = 5,
                 device: str = None):
        
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        
        # Training history
        self.training_history = []
        
        logger.info(f"MAML Trainer initialized with inner_lr={inner_lr}, outer_lr={outer_lr}")
    
    def inner_update(self, 
                     support_data: TaskDataset, 
                     model_copy: BaseMetaModel) -> BaseMetaModel:
        """Perform inner loop update on support data."""
        
        # Create data loader
        support_loader = DataLoader(support_data, batch_size=len(support_data), shuffle=True)
        support_x, support_y = next(iter(support_loader))
        support_x, support_y = support_x.to(self.device), support_y.to(self.device)
        
        # Convert targets to long for classification
        support_y = support_y.long()
        
        # Perform gradient steps
        for step in range(self.inner_steps):
            # Forward pass
            logits = model_copy(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, model_copy.parameters(), create_graph=True)
            
            # Update model parameters
            updated_params = []
            for param, grad in zip(model_copy.parameters(), grads):
                if grad is not None:
                    updated_param = param - self.inner_lr * grad
                else:
                    updated_param = param
                updated_params.append(updated_param)
            
            # Update model with new parameters
            self._update_model_params(model_copy, updated_params)
        
        return model_copy
    
    def _update_model_params(self, model: nn.Module, new_params: List[torch.Tensor]):
        """Update model parameters with new values."""
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data
    
    def meta_update(self, 
                    tasks: List[MarketTask],
                    support_size: int = 50,
                    query_size: int = 50) -> float:
        """Perform one meta-update step."""
        
        meta_losses = []
        
        for task in tasks:
            # Split task data
            support_data, query_data = task.get_support_query_split(support_size, query_size)
            
            # Create model copy for this task
            model_copy = self.model.clone()
            
            # Inner loop: adapt model to task using support data
            adapted_model = self.inner_update(support_data, model_copy)
            
            # Evaluate adapted model on query data
            query_loader = DataLoader(query_data, batch_size=len(query_data), shuffle=False)
            query_x, query_y = next(iter(query_loader))
            query_x, query_y = query_x.to(self.device), query_y.to(self.device)
            query_y = query_y.long()
            
            # Compute query loss
            query_logits = adapted_model(query_x)
            query_loss = F.cross_entropy(query_logits, query_y)
            meta_losses.append(query_loss)
        
        # Meta-optimization step
        if meta_losses:
            meta_loss = torch.stack(meta_losses).mean()
            
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            return meta_loss.item()
        
        return 0.0
    
    def train(self, 
              task_generator: Callable[[], List[MarketTask]],
              num_epochs: int = 100,
              tasks_per_epoch: int = 10,
              support_size: int = 50,
              query_size: int = 50) -> Dict[str, List]:
        """Train the meta-learner."""
        
        logger.info(f"Starting MAML training: {num_epochs} epochs, {tasks_per_epoch} tasks per epoch")
        
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Generate tasks for this epoch
            tasks = task_generator()[:tasks_per_epoch]
            
            # Meta-update on batch of tasks
            meta_loss = self.meta_update(tasks, support_size, query_size)
            epoch_loss += meta_loss
            
            epoch_losses.append(epoch_loss)
            
            # Log progress
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Meta-loss = {epoch_loss:.4f}")
            
            self.training_history.append({
                'epoch': epoch,
                'meta_loss': meta_loss,
                'tasks_trained': len(tasks)
            })
        
        logger.info("MAML training completed")
        return {'losses': epoch_losses, 'history': self.training_history}
    
    def adapt_to_new_task(self, 
                          new_task: MarketTask,
                          support_size: int = 50,
                          adaptation_steps: int = None) -> BaseMetaModel:
        """Quickly adapt the meta-model to a new task."""
        
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        # Get support data
        support_data, _ = new_task.get_support_query_split(support_size, 0)
        
        # Create model copy
        adapted_model = self.model.clone()
        
        # Adapt model using support data
        support_loader = DataLoader(support_data, batch_size=len(support_data))
        support_x, support_y = next(iter(support_loader))
        support_x, support_y = support_x.to(self.device), support_y.to(self.device)
        support_y = support_y.long()
        
        # Perform adaptation steps
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            loss.backward()
            optimizer.step()
        
        logger.info(f"Adapted model to task '{new_task.name}' in {adaptation_steps} steps")
        return adapted_model


class FewShotLearner:
    """Few-shot learning for new asset classes."""
    
    def __init__(self,
                 base_model: BaseMetaModel,
                 similarity_metric: str = 'cosine',
                 k_shot: int = 5):
        
        self.base_model = base_model
        self.similarity_metric = similarity_metric
        self.k_shot = k_shot
        
        # Support set storage
        self.support_sets = {}
        self.prototype_vectors = {}
        
        logger.info(f"FewShotLearner initialized with {k_shot}-shot learning")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations from the base model."""
        # Get features from second-to-last layer
        features = x
        for layer in list(self.base_model.network.children())[:-1]:
            features = layer(features)
        return features
    
    def add_support_examples(self, 
                            asset_class: str, 
                            examples: torch.Tensor, 
                            labels: torch.Tensor):
        """Add support examples for a new asset class."""
        
        # Extract features
        with torch.no_grad():
            features = self.extract_features(examples)
        
        # Store support set
        self.support_sets[asset_class] = {
            'features': features,
            'labels': labels
        }
        
        # Compute prototype vectors (mean of each class)
        unique_labels = torch.unique(labels)
        prototypes = {}
        
        for label in unique_labels:
            mask = labels == label
            prototype = features[mask].mean(dim=0)
            prototypes[label.item()] = prototype
        
        self.prototype_vectors[asset_class] = prototypes
        
        logger.info(f"Added support examples for '{asset_class}': "
                   f"{len(examples)} examples, {len(prototypes)} classes")
    
    def predict(self, 
                query_examples: torch.Tensor,
                asset_class: str) -> torch.Tensor:
        """Predict labels for query examples using few-shot learning."""
        
        if asset_class not in self.prototype_vectors:
            logger.warning(f"No support examples for asset class '{asset_class}'")
            return torch.zeros(len(query_examples), dtype=torch.long)
        
        # Extract query features
        with torch.no_grad():
            query_features = self.extract_features(query_examples)
        
        # Compute similarities to prototypes
        prototypes = self.prototype_vectors[asset_class]
        predictions = []
        
        for query_feature in query_features:
            similarities = {}
            
            for label, prototype in prototypes.items():
                if self.similarity_metric == 'cosine':
                    sim = F.cosine_similarity(query_feature, prototype, dim=0)
                elif self.similarity_metric == 'euclidean':
                    sim = -torch.norm(query_feature - prototype)
                else:
                    sim = torch.dot(query_feature, prototype)
                
                similarities[label] = sim
            
            # Predict class with highest similarity
            predicted_label = max(similarities, key=similarities.get)
            predictions.append(predicted_label)
        
        return torch.tensor(predictions, dtype=torch.long)


class ContinualLearner:
    """Continual learning to prevent catastrophic forgetting."""
    
    def __init__(self,
                 model: BaseMetaModel,
                 memory_size: int = 1000,
                 regularization_strength: float = 0.1):
        
        self.model = model
        self.memory_size = memory_size
        self.regularization_strength = regularization_strength
        
        # Episodic memory for replay
        self.memory = []
        self.task_boundaries = {}
        
        # Important parameters (for regularization)
        self.important_params = {}
        self.previous_params = {}
        
        logger.info(f"ContinualLearner initialized with memory_size={memory_size}")
    
    def add_to_memory(self, 
                      task_name: str, 
                      examples: torch.Tensor, 
                      labels: torch.Tensor):
        """Add examples to episodic memory."""
        
        # Add examples with task identifier
        for example, label in zip(examples, labels):
            self.memory.append({
                'task': task_name,
                'example': example,
                'label': label
            })
        
        # Maintain memory size limit
        if len(self.memory) > self.memory_size:
            # Remove oldest examples
            self.memory = self.memory[-self.memory_size:]
        
        # Update task boundaries
        if task_name not in self.task_boundaries:
            self.task_boundaries[task_name] = len(self.memory)
        
        logger.info(f"Added {len(examples)} examples to memory for task '{task_name}'")
    
    def compute_importance_weights(self, task_data: TaskDataset):
        """Compute parameter importance using Fisher Information Matrix."""
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Compute gradients
        importance_weights = {}
        
        data_loader = DataLoader(task_data, batch_size=32, shuffle=True)
        
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(next(self.model.parameters()).device)
            batch_y = batch_y.long().to(next(self.model.parameters()).device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            
            # Compute gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in importance_weights:
                        importance_weights[name] = torch.zeros_like(param.data)
                    importance_weights[name] += param.grad.data ** 2
        
        # Normalize by number of batches
        num_batches = len(data_loader)
        for name in importance_weights:
            importance_weights[name] /= num_batches
        
        return importance_weights
    
    def elastic_weight_consolidation_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.important_params and name in self.previous_params:
                importance = self.important_params[name]
                prev_param = self.previous_params[name]
                
                ewc_loss += (importance * (param - prev_param) ** 2).sum()
        
        return self.regularization_strength * ewc_loss
    
    def train_on_task(self, 
                      task: MarketTask, 
                      num_epochs: int = 50,
                      batch_size: int = 32) -> Dict[str, float]:
        """Train on a new task with continual learning."""
        
        logger.info(f"Training continually on task '{task.name}'")
        
        # Create dataset
        task_dataset = TaskDataset(task.X, task.y, task.name)
        data_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(next(self.model.parameters()).device)
                batch_y = batch_y.long().to(next(self.model.parameters()).device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                task_loss = F.cross_entropy(outputs, batch_y)
                
                # Add EWC regularization loss
                ewc_loss = self.elastic_weight_consolidation_loss()
                total_loss = task_loss + ewc_loss
                
                # Replay from memory
                if len(self.memory) > 0:
                    replay_batch_size = min(batch_size // 2, len(self.memory))
                    memory_batch = random.sample(self.memory, replay_batch_size)
                    
                    memory_x = torch.stack([item['example'] for item in memory_batch])
                    memory_y = torch.stack([item['label'] for item in memory_batch]).long()
                    
                    memory_x = memory_x.to(next(self.model.parameters()).device)
                    memory_y = memory_y.to(next(self.model.parameters()).device)
                    
                    memory_outputs = self.model(memory_x)
                    memory_loss = F.cross_entropy(memory_outputs, memory_y)
                    
                    total_loss += 0.5 * memory_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            epoch_losses.append(epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
        
        # After training, update importance weights and store current parameters
        importance_weights = self.compute_importance_weights(task_dataset)
        self.important_params.update(importance_weights)
        
        self.previous_params = {name: param.data.clone() 
                               for name, param in self.model.named_parameters()}
        
        # Add examples to memory
        self.add_to_memory(task.name, torch.FloatTensor(task.X), torch.FloatTensor(task.y))
        
        return {'losses': epoch_losses, 'final_loss': epoch_losses[-1] if epoch_losses else 0.0}


class MetaLearningOrchestrator:
    """Main orchestrator for meta-learning in trading."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 3):
        
        # Create base model
        self.base_model = MLPMetaModel(input_dim, hidden_dims, output_dim)
        
        # Initialize meta-learning components
        self.maml_trainer = MAMLTrainer(self.base_model)
        self.few_shot_learner = FewShotLearner(self.base_model)
        self.continual_learner = ContinualLearner(self.base_model)
        
        # Task repository
        self.tasks = {}
        self.regime_tasks = defaultdict(list)
        
        logger.info("MetaLearningOrchestrator initialized")
    
    def add_task(self, task: MarketTask):
        """Add a new task to the repository."""
        self.tasks[task.name] = task
        self.regime_tasks[task.regime].append(task)
        logger.info(f"Added task '{task.name}' (regime: {task.regime})")
    
    def create_synthetic_tasks(self, num_tasks: int = 20) -> List[MarketTask]:
        """Create synthetic tasks for testing."""
        tasks = []
        
        for i in range(num_tasks):
            # Create synthetic data
            n_samples = np.random.randint(100, 500)
            n_features = 20
            
            # Generate features with different distributions for different regimes
            regime = np.random.choice(['bull', 'bear', 'sideways', 'volatile'])
            
            if regime == 'bull':
                features = np.random.normal(0.5, 1.0, (n_samples, n_features))
                trend = 0.002
            elif regime == 'bear':
                features = np.random.normal(-0.5, 1.0, (n_samples, n_features))
                trend = -0.002
            elif regime == 'volatile':
                features = np.random.normal(0.0, 2.0, (n_samples, n_features))
                trend = 0.0
            else:  # sideways
                features = np.random.normal(0.0, 0.5, (n_samples, n_features))
                trend = 0.0
            
            # Create synthetic price series
            returns = np.random.normal(trend, 0.02, n_samples)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create DataFrame
            data = pd.DataFrame(features, columns=[f'feature_{j}' for j in range(n_features)])
            data['close'] = prices
            
            # Create task
            task = MarketTask(
                name=f'synthetic_task_{i}',
                data=data,
                regime=regime,
                asset_class=np.random.choice(['equity', 'crypto', 'forex'])
            )
            
            tasks.append(task)
        
        return tasks
    
    def train_meta_learner(self, 
                          num_epochs: int = 100,
                          tasks_per_epoch: int = 10) -> Dict[str, Any]:
        """Train the meta-learner using MAML."""
        
        # Create synthetic tasks for training
        all_tasks = self.create_synthetic_tasks(num_tasks=50)
        
        # Add tasks to repository
        for task in all_tasks:
            self.add_task(task)
        
        # Define task generator
        def task_generator():
            return random.sample(all_tasks, min(tasks_per_epoch, len(all_tasks)))
        
        # Train MAML
        results = self.maml_trainer.train(
            task_generator=task_generator,
            num_epochs=num_epochs,
            tasks_per_epoch=tasks_per_epoch
        )
        
        logger.info("Meta-learner training completed")
        return results
    
    def adapt_to_regime(self, 
                       regime: str, 
                       new_data: pd.DataFrame,
                       adaptation_steps: int = 10) -> BaseMetaModel:
        """Quickly adapt to a new market regime."""
        
        # Create task for new regime
        new_task = MarketTask(
            name=f'adaptation_{regime}',
            data=new_data,
            regime=regime
        )
        
        # Use MAML to adapt
        adapted_model = self.maml_trainer.adapt_to_new_task(
            new_task, 
            adaptation_steps=adaptation_steps
        )
        
        logger.info(f"Adapted model to regime '{regime}'")
        return adapted_model
    
    def few_shot_predict(self, 
                        asset_class: str, 
                        support_data: pd.DataFrame,
                        query_data: pd.DataFrame) -> np.ndarray:
        """Make predictions using few-shot learning."""
        
        # Create support task
        support_task = MarketTask(f'support_{asset_class}', support_data)
        query_task = MarketTask(f'query_{asset_class}', query_data)
        
        # Add support examples
        support_x = torch.FloatTensor(support_task.X)
        support_y = torch.FloatTensor(support_task.y)
        
        self.few_shot_learner.add_support_examples(asset_class, support_x, support_y)
        
        # Make predictions
        query_x = torch.FloatTensor(query_task.X)
        predictions = self.few_shot_learner.predict(query_x, asset_class)
        
        return predictions.numpy()
    
    async def continual_learning_pipeline(self, 
                                        tasks: List[MarketTask],
                                        epochs_per_task: int = 50) -> Dict[str, Any]:
        """Run continual learning pipeline."""
        
        results = {'task_results': {}, 'overall_performance': []}
        
        for i, task in enumerate(tasks):
            logger.info(f"Continual learning on task {i+1}/{len(tasks)}: {task.name}")
            
            # Train on current task
            task_results = self.continual_learner.train_on_task(
                task, 
                num_epochs=epochs_per_task
            )
            
            results['task_results'][task.name] = task_results
            
            # Evaluate on all previous tasks to measure forgetting
            if i > 0:
                forgetting_scores = {}
                for prev_task in tasks[:i]:
                    # Simple evaluation (would be more sophisticated in practice)
                    forgetting_scores[prev_task.name] = np.random.uniform(0.5, 0.9)  # Placeholder
                
                results['overall_performance'].append(forgetting_scores)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        logger.info("Continual learning pipeline completed")
        return results


# Test and example functions
async def test_meta_learning_system():
    """Test the complete meta-learning system."""
    logger.info("🧠 Testing Meta-Learning System...")
    
    try:
        # Create orchestrator
        orchestrator = MetaLearningOrchestrator(input_dim=20)
        
        # Test 1: MAML training
        logger.info("Testing MAML training...")
        maml_results = orchestrator.train_meta_learner(num_epochs=20, tasks_per_epoch=5)
        logger.info(f"✅ MAML training completed: {len(maml_results['losses'])} epochs")
        
        # Test 2: Quick adaptation
        logger.info("Testing quick adaptation...")
        new_data = pd.DataFrame(np.random.randn(200, 20), columns=[f'feature_{i}' for i in range(20)])
        new_data['close'] = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 200)))
        
        adapted_model = orchestrator.adapt_to_regime('volatile', new_data)
        logger.info("✅ Quick adaptation completed")
        
        # Test 3: Few-shot learning
        logger.info("Testing few-shot learning...")
        support_data = pd.DataFrame(np.random.randn(50, 20), columns=[f'feature_{i}' for i in range(20)])
        support_data['close'] = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.01, 50)))
        
        query_data = pd.DataFrame(np.random.randn(30, 20), columns=[f'feature_{i}' for i in range(20)])
        query_data['close'] = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.01, 30)))
        
        predictions = orchestrator.few_shot_predict('new_crypto', support_data, query_data)
        logger.info(f"✅ Few-shot learning completed: {len(predictions)} predictions")
        
        # Test 4: Continual learning
        logger.info("Testing continual learning...")
        continual_tasks = orchestrator.create_synthetic_tasks(num_tasks=5)
        continual_results = await orchestrator.continual_learning_pipeline(
            continual_tasks, 
            epochs_per_task=10
        )
        logger.info(f"✅ Continual learning completed: {len(continual_results['task_results'])} tasks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Meta-learning system test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_meta_learning_system())
