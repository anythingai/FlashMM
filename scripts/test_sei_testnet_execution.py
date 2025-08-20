"""
Sei V2 Testnet Order Execution Test

IMPORTANT: This script will execute real orders on Sei V2 testnet.
Only run this script when you have:
1. Valid Sei testnet private key with test tokens
2. Valid Cambrian API credentials for testnet
3. Sufficient test SEI and USDC tokens

This script tests actual order execution on the blockchain.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from decimal import Decimal
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from flashmm.blockchain.blockchain_service import get_blockchain_service
from flashmm.trading.execution.order_router import OrderRouter
from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)


class SeiTestnetExecutor:
    """Executes real orders on Sei V2 testnet for validation."""
    
    def __init__(self):
        self.config = get_config()
        self.blockchain_service = None
        self.order_router = None
        self.test_results = {}
        
        # Test configuration - SAFE defaults to minimize risk
        self.test_orders = [
            {
                "name": "small_buy_order",
                "symbol": "SEI/USDC",
                "side": "buy",
                "price": 0.01,      # Very low price to avoid execution
                "size": 1.0,        # Very small size
                "order_type": "limit",
                "description": "Small buy order at low price"
            },
            {
                "name": "small_sell_order", 
                "symbol": "SEI/USDC",
                "side": "sell",
                "price": 100.0,     # Very high price to avoid execution
                "size": 1.0,        # Very small size
                "order_type": "limit",
                "description": "Small sell order at high price"
            }
        ]
    
    def _check_testnet_configuration(self) -> Dict[str, Any]:
        """Check if testnet configuration is properly set."""
        config_check = {
            'sei_private_key': bool(os.getenv('SEI_PRIVATE_KEY')),
            'cambrian_api_key': bool(os.getenv('CAMBRIAN_API_KEY')),
            'cambrian_secret_key': bool(os.getenv('CAMBRIAN_SECRET_KEY')),
            'blockchain_enabled': self.config.get('blockchain.enabled', False)
        }
        
        config_check['all_configured'] = all(config_check.values())
        
        return config_check
    
    async def run_testnet_execution_test(self) -> Dict[str, Any]:
        """Run testnet execution test."""
        logger.info("=== SEI V2 TESTNET EXECUTION TEST ===")
        logger.warning("THIS WILL EXECUTE REAL ORDERS ON TESTNET")
        
        try:
            # 1. Check configuration
            config_check = self._check_testnet_configuration()
            self.test_results['configuration_check'] = config_check
            
            if not config_check['all_configured']:
                logger.error("❌ Testnet configuration incomplete")
                missing = [k for k, v in config_check.items() if not v and k != 'all_configured']
                logger.error(f"Missing: {missing}")
                return self.test_results
            
            logger.info("✅ Testnet configuration verified")
            
            # 2. Initialize services
            await self._initialize_services()
            
            # 3. Check network connectivity
            await self._test_network_connectivity()
            
            # 4. Check account balances
            await self._check_account_balances()
            
            # 5. Execute test orders
            await self._execute_test_orders()
            
            # 6. Test order management
            await self._test_order_management()
            
            # 7. Generate final report
            return self._generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Testnet execution test failed: {e}")
            self.test_results['fatal_error'] = str(e)
            return self.test_results
        
        finally:
            await self._cleanup()
    
    async def _initialize_services(self):
        """Initialize blockchain services."""
        logger.info("Initializing blockchain services...")
        
        try:
            # Initialize blockchain service
            self.blockchain_service = get_blockchain_service()
            await self.blockchain_service.initialize()
            
            # Initialize order router
            self.order_router = OrderRouter()
            await self.order_router.initialize()
            
            # Check readiness
            service_status = self.blockchain_service.get_service_status()
            is_ready = self.order_router.is_blockchain_ready()
            
            self.test_results['service_initialization'] = {
                'success': True,
                'blockchain_service_status': service_status.get('status'),
                'order_router_ready': is_ready,
                'supported_markets': self.blockchain_service.get_supported_markets()
            }
            
            logger.info("✅ Services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            self.test_results['service_initialization'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    async def _test_network_connectivity(self):
        """Test network connectivity to Sei testnet."""
        logger.info("Testing network connectivity...")
        
        try:
            network_status = self.blockchain_service.get_network_status()
            
            # Force a network health check
            health_result = await self.blockchain_service.force_health_check()
            
            connectivity_success = (
                network_status.get('status') in ['healthy', 'degraded'] and
                'block_height' in network_status and
                network_status['block_height'] > 0
            )
            
            self.test_results['network_connectivity'] = {
                'success': connectivity_success,
                'network_status': network_status,
                'health_check_result': health_result,
                'block_height': network_status.get('block_height', 0)
            }
            
            if connectivity_success:
                logger.info(f"✅ Network connectivity verified - Block height: {network_status.get('block_height')}")
            else:
                logger.error("❌ Network connectivity issues detected")
                
        except Exception as e:
            logger.error(f"❌ Network connectivity test failed: {e}")
            self.test_results['network_connectivity'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    async def _check_account_balances(self):
        """Check account balances before trading."""
        logger.info("Checking account balances...")
        
        try:
            balance_info = {}
            
            if self.blockchain_service.account_manager:
                try:
                    active_account = await self.blockchain_service.account_manager.get_active_account()
                    if active_account:
                        balance = await self.blockchain_service.account_manager.get_account_balance(active_account.account_id)
                        if balance:
                            balance_info = {
                                'address': balance.address,
                                'balances': {k: float(v) for k, v in balance.balances.items()},
                                'total_value_usdc': float(balance.total_value_usdc)
                            }
                except Exception as e:
                    logger.warning(f"Balance query failed: {e}")
                    balance_info = {'error': str(e)}
            
            self.test_results['account_balances'] = {
                'success': True,
                'balance_info': balance_info
            }
            
            if balance_info and 'balances' in balance_info:
                logger.info(f"✅ Account balances retrieved: {balance_info['balances']}")
            else:
                logger.warning("⚠️  Could not retrieve account balances")
                
        except Exception as e:
            logger.error(f"❌ Balance check failed: {e}")
            self.test_results['account_balances'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_test_orders(self):
        """Execute test orders on testnet."""
        logger.info("Executing test orders...")
        logger.warning("🚨 PLACING REAL ORDERS ON TESTNET - ENSURE YOU HAVE TEST TOKENS")
        
        order_results = []
        
        for test_order in self.test_orders:
            logger.info(f"Placing {test_order['description']}...")
            
            try:
                # Place order
                order = await self.order_router.place_order(
                    symbol=test_order['symbol'],
                    side=test_order['side'],
                    price=test_order['price'],
                    size=test_order['size'],
                    order_type=test_order['order_type'],
                    time_in_force="GTC"
                )
                
                if order:
                    order_result = {
                        'test_name': test_order['name'],
                        'success': True,
                        'order_id': order.order_id,
                        'symbol': order.symbol,
                        'side': order.side,
                        'price': float(order.price),
                        'size': float(order.size),
                        'status': order.status.value,
                        'created_at': order.created_at.isoformat()
                    }
                    
                    logger.info(f"✅ Order placed: {order.order_id} - {order.symbol} {order.side} {order.size}@{order.price}")
                    
                    # Wait a moment for processing
                    await asyncio.sleep(2)
                    
                    # Check order status
                    updated_order = self.order_router.get_order(order.order_id)
                    if updated_order:
                        order_result['final_status'] = updated_order.status.value
                        logger.info(f"Order status: {updated_order.status.value}")
                    
                else:
                    order_result = {
                        'test_name': test_order['name'],
                        'success': False,
                        'error': 'Order placement returned None'
                    }
                    logger.error(f"❌ Order placement failed for {test_order['name']}")
                
                order_results.append(order_result)
                
            except Exception as e:
                order_result = {
                    'test_name': test_order['name'],
                    'success': False,
                    'error': str(e)
                }
                order_results.append(order_result)
                logger.error(f"❌ Order placement error for {test_order['name']}: {e}")
        
        self.test_results['order_execution'] = {
            'success': any(result['success'] for result in order_results),
            'total_orders': len(self.test_orders),
            'successful_orders': sum(1 for result in order_results if result['success']),
            'order_results': order_results
        }
    
    async def _test_order_management(self):
        """Test order management operations."""
        logger.info("Testing order management...")
        
        management_results = {
            'order_queries': False,
            'order_cancellations': False,
            'performance_stats': False
        }
        
        try:
            # Test order queries
            all_orders = [order_id for order_id in self.order_router.orders.keys()]
            active_orders = self.order_router.get_active_orders()
            
            management_results['order_queries'] = True
            logger.info(f"✅ Order queries successful - {len(all_orders)} total, {len(active_orders)} active")
            
            # Test order cancellations
            cancelled_count = 0
            for order_id in all_orders:
                try:
                    success = await self.order_router.cancel_order(order_id)
                    if success:
                        cancelled_count += 1
                        logger.info(f"✅ Cancelled order: {order_id}")
                except Exception as e:
                    logger.warning(f"Cancel failed for {order_id}: {e}")
            
            management_results['order_cancellations'] = cancelled_count > 0
            
            # Test performance stats
            perf_stats = self.order_router.get_performance_stats()
            management_results['performance_stats'] = 'blockchain_integration' in perf_stats
            
            logger.info(f"✅ Order management tests completed - {cancelled_count} orders cancelled")
            
        except Exception as e:
            logger.error(f"❌ Order management test failed: {e}")
        
        self.test_results['order_management'] = {
            'success': any(management_results.values()),
            'management_results': management_results
        }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        # Count successful tests
        successful_tests = sum(
            1 for result in self.test_results.values()
            if isinstance(result, dict) and result.get('success', False)
        )
        
        total_tests = len([k for k in self.test_results.keys() if k != 'fatal_error'])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        overall_success = (
            'fatal_error' not in self.test_results and
            success_rate >= 80 and
            self.test_results.get('order_execution', {}).get('success', False)
        )
        
        report = {
            'test_summary': {
                'overall_success': overall_success,
                'success_rate_percent': round(success_rate, 1),
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'orders_executed': self.test_results.get('order_execution', {}).get('successful_orders', 0),
                'testnet_ready': overall_success
            },
            'detailed_results': self.test_results,
            'recommendations': self._generate_testnet_recommendations()
        }
        
        return report
    
    def _generate_testnet_recommendations(self) -> List[str]:
        """Generate recommendations based on testnet results."""
        recommendations = []
        
        # Check configuration
        config_check = self.test_results.get('configuration_check', {})
        if not config_check.get('all_configured'):
            recommendations.append("Complete testnet configuration with all required keys")
        
        # Check network connectivity
        network_test = self.test_results.get('network_connectivity', {})
        if not network_test.get('success'):
            recommendations.append("Verify network connectivity to Sei V2 testnet")
        
        # Check order execution
        order_execution = self.test_results.get('order_execution', {})
        if not order_execution.get('success'):
            recommendations.append("Debug order execution issues before production deployment")
        
        # General recommendations
        recommendations.extend([
            "Verify sufficient test token balances before extensive testing",
            "Monitor gas costs and transaction fees during testing",
            "Test order execution during different network conditions",
            "Validate emergency stop procedures work correctly",
            "Test with different order sizes and market conditions"
        ])
        
        return recommendations
    
    async def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.order_router:
                # Cancel any remaining orders
                active_orders = self.order_router.get_active_orders()
                if active_orders:
                    logger.info(f"Cancelling {len(active_orders)} remaining orders...")
                    await self.order_router.cancel_all_orders()
                
                await self.order_router.cleanup()
            
            if self.blockchain_service:
                await self.blockchain_service.cleanup()
                
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def main():
    """Main execution function."""
    print("🚨 WARNING: This script will execute REAL orders on Sei V2 testnet!")
    print("Make sure you have:")
    print("1. Valid testnet private key with test tokens")
    print("2. Valid Cambrian API credentials") 
    print("3. Sufficient SEI and USDC test tokens")
    print()
    
    # Safety check
    confirm = input("Do you want to proceed? (type 'YES' to continue): ")
    if confirm != 'YES':
        print("Test cancelled by user")
        return 0
    
    executor = SeiTestnetExecutor()
    
    try:
        # Run testnet execution test
        report = await executor.run_testnet_execution_test()
        
        # Print results
        print("\n" + "="*60)
        print("SEI V2 TESTNET EXECUTION TEST REPORT")
        print("="*60)
        
        summary = report['test_summary']
        print(f"Overall Success: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        print(f"Success Rate: {summary['success_rate_percent']}%")
        print(f"Tests: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"Orders Executed: {summary['orders_executed']}")
        print(f"Testnet Ready: {'✅ YES' if summary['testnet_ready'] else '❌ NO'}")
        
        print("\nDetailed Results:")
        print("-" * 30)
        
        for test_name, result in report['detailed_results'].items():
            if test_name == 'fatal_error':
                print(f"❌ FATAL ERROR: {result}")
                continue
                
            status = "✅ PASS" if result.get('success') else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
            
            if not result.get('success') and 'error' in result:
                print(f"    Error: {result['error']}")
        
        # Show order execution details
        order_execution = report['detailed_results'].get('order_execution', {})
        if 'order_results' in order_execution:
            print("\nOrder Execution Details:")
            print("-" * 25)
            for order_result in order_execution['order_results']:
                status = "✅" if order_result['success'] else "❌"
                print(f"{status} {order_result['test_name']}")
                if order_result['success']:
                    print(f"    Order ID: {order_result.get('order_id', 'N/A')}")
                    print(f"    Status: {order_result.get('final_status', order_result.get('status', 'N/A'))}")
        
        print("\nRecommendations:")
        print("-" * 20)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)
        
        # Save detailed report
        report_file = project_root / "testnet_execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Detailed report saved to: {report_file}")
        
        return 0 if summary['overall_success'] else 1
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)