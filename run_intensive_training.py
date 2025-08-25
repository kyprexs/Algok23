"""
Intensive Training Script for AgloK23 Advanced ML Pipeline

This script runs continuous training sessions to improve model performance
and test various configurations across different market regimes.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.training.continuous_trainer import AdvancedEnsembleTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_logs.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def run_mega_training_session():
    """
    Run an extensive training session with multiple phases
    """
    print("🚀 Starting MEGA Training Session for AgloK23!")
    print("=" * 80)
    
    # Initialize the advanced trainer
    trainer = AdvancedEnsembleTrainer()
    
    total_start_time = datetime.now()
    
    try:
        # Phase 1: Initial Intensive Training
        print("🔥 PHASE 1: Intensive Training Session")
        print("-" * 50)
        
        intensive_results = await trainer.intensive_training_session(
            training_rounds=15,  # More rounds for better coverage
            samples_per_round=2000,  # More samples per round
            market_regimes=['bull', 'bear', 'sideways', 'volatile', 'normal', 'crash']  # Added crash regime
        )
        
        print(f"✅ Phase 1 completed! Best accuracy: {intensive_results['best_models']['overall']['info']['metrics']['accuracy']:.4f}")
        print(f"📊 Regime performance:")
        for regime, stats in intensive_results['regime_performance'].items():
            print(f"  {regime}: avg={stats['avg_accuracy']:.4f}, best={stats['best_accuracy']:.4f}")
        
        # Phase 2: Continuous Improvement
        print("\n🔧 PHASE 2: Continuous Improvement Loop")
        print("-" * 50)
        
        improvement_results = await trainer.continuous_improvement_loop(
            improvement_rounds=10,  # More improvement rounds
            focus_areas=[
                'hyperparameters', 'architecture', 'features', 'ensembling', 
                'data_augmentation', 'hyperparameters', 'architecture', 'features',
                'ensembling', 'data_augmentation'  # Double cycle for thoroughness
            ]
        )
        
        print("✅ Phase 2 completed! Improvement results:")
        for round_name, results in improvement_results.items():
            if 'best_score' in results:
                print(f"  {round_name}: {results['best_score']:.4f}")
        
        # Phase 3: Advanced Optimization
        print("\n⚡ PHASE 3: Advanced Optimization")
        print("-" * 50)
        
        # Run additional focused training on best configurations
        best_configs = []
        
        # Collect best configurations from improvement phase
        for round_name, results in improvement_results.items():
            if 'best_config' in results and results['best_config']:
                best_configs.append(results['best_config'])
            if 'best_arch' in results and results['best_arch']:
                best_configs.append(results['best_arch'])
            if 'best_weights' in results and results['best_weights']:
                best_configs.append(results['best_weights'])
        
        # Run final optimization with best configurations
        final_results = []
        for i, config in enumerate(best_configs[:5]):  # Test top 5 configurations
            print(f"🎯 Testing optimal configuration {i+1}/5...")
            
            # Apply configuration
            if 'sequence_length' in config:
                trainer.base_config['sequence_length'] = config['sequence_length']
            if 'lstm_layers' in config:
                trainer.base_config['lstm_config']['num_layers'] = config['lstm_layers']
            if 'transformer_layers' in config:
                trainer.base_config['transformer_config']['num_layers'] = config['transformer_layers']
            
            # Create high-quality diverse dataset
            test_df = trainer.create_synthetic_financial_data(
                n_samples=3000,
                n_features=40,
                market_regime='volatile'  # Most challenging regime
            )
            
            feature_columns = [col for col in test_df.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            from src.core.models.advanced_ensemble import AdvancedEnsemble
            ensemble = AdvancedEnsemble(**trainer.base_config)
            ensemble.optimizer.n_trials = 200  # Maximum optimization
            ensemble.optimizer.timeout = 3600   # 1 hour per config
            
            results = await ensemble.train_ensemble(test_df, feature_columns, validation_split=0.2)
            final_results.append({
                'config_index': i,
                'config': config,
                'accuracy': results['ensemble_accuracy'],
                'individual_performance': results['individual_performance']
            })
            
            print(f"  Configuration {i+1} accuracy: {results['ensemble_accuracy']:.4f}")
        
        # Find absolute best configuration
        best_final = max(final_results, key=lambda x: x['accuracy'])
        
        print("\n🏆 FINAL RESULTS")
        print("=" * 80)
        
        total_time = (datetime.now() - total_start_time).total_seconds()
        
        print(f"⏱️  Total training time: {total_time/3600:.1f} hours")
        print(f"🎯 Best overall accuracy: {best_final['accuracy']:.4f}")
        print(f"🏅 Best configuration: {best_final['config']}")
        print(f"📈 Individual model performance:")
        for model, acc in best_final['individual_performance'].items():
            print(f"  {model}: {acc:.4f}")
        
        # Generate comprehensive summary
        training_summary = trainer.get_training_summary()
        print(f"\n📊 Training Summary:")
        print(f"  Total models trained: {training_summary['total_models_trained']}")
        print(f"  Best performer: {training_summary['best_performers']['overall']['model']}")
        print(f"  Best score: {training_summary['best_performers']['overall']['score']:.4f}")
        
        print("\n🎓 Training Insights:")
        for insight in training_summary['training_insights']:
            print(f"  {insight}")
        
        # Save detailed results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_training_time_hours': total_time / 3600,
            'intensive_training': intensive_results,
            'improvement_results': improvement_results,
            'final_optimization': final_results,
            'best_configuration': best_final,
            'training_summary': training_summary
        }
        
        with open(f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"❌ Error during mega training session: {e}")
        raise


async def run_quick_improvement_test():
    """
    Run a quick test of the improvement system
    """
    print("⚡ Running Quick Improvement Test")
    print("=" * 50)
    
    trainer = AdvancedEnsembleTrainer()
    
    # Quick intensive session
    results = await trainer.intensive_training_session(
        training_rounds=5,
        samples_per_round=1000,
        market_regimes=['bull', 'bear', 'volatile']
    )
    
    print(f"✅ Quick test completed!")
    print(f"🎯 Best accuracy: {results['best_models']['overall']['info']['metrics']['accuracy']:.4f}")
    print(f"⏱️  Training time: {results['total_training_time']/60:.1f} minutes")
    
    return results


async def main():
    """
    Main training orchestrator
    """
    print("🤖 AgloK23 Advanced ML Training System")
    print("=" * 80)
    
    # Ask user for training intensity
    print("Choose training mode:")
    print("1. 🔥 MEGA Training Session (5-8 hours, maximum performance)")
    print("2. ⚡ Quick Improvement Test (30-60 minutes, basic testing)")
    print("3. 🎯 Custom Training Session")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except KeyboardInterrupt:
        choice = '2'  # Default to quick test
    
    if choice == '1':
        print("\n🔥 Starting MEGA Training Session...")
        print("⚠️  Warning: This will take 5-8 hours and use significant computational resources!")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            results = await run_mega_training_session()
        else:
            print("Training cancelled.")
            return
    
    elif choice == '2':
        print("\n⚡ Starting Quick Improvement Test...")
        results = await run_quick_improvement_test()
    
    elif choice == '3':
        print("\n🎯 Custom Training Session")
        try:
            rounds = int(input("Training rounds (5-20): ") or "10")
            samples = int(input("Samples per round (1000-3000): ") or "1500")
            
            trainer = AdvancedEnsembleTrainer()
            results = await trainer.intensive_training_session(
                training_rounds=rounds,
                samples_per_round=samples
            )
            
            print(f"✅ Custom training completed!")
            print(f"🎯 Best accuracy: {results['best_models']['overall']['info']['metrics']['accuracy']:.4f}")
        
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled. Running quick test instead...")
            results = await run_quick_improvement_test()
    
    else:
        print("Invalid choice. Running quick test...")
        results = await run_quick_improvement_test()
    
    print("\n🎉 Training session completed successfully!")
    print("🚀 Your AgloK23 ML models are now more powerful!")


if __name__ == "__main__":
    asyncio.run(main())
