"""
Tests for FleeingTargetStrategy - target movement away from drone with obstacle avoidance.
"""
import numpy as np
import pytest

from agents.drone import Drone
from utils.target_strategies import FleeingTargetStrategy
from utils.generation import EnvData
from config.env_config import (
    TARGET_MAX_SPEED,
    TARGET_MAX_ANGULAR_VELOCITY,
    TARGET_MIN_DISTANCE_TO_OBSTACLE,
    DELTA_T,
    FIELD_SIZE,
    OBSTACLE_COLLISION_MAX_RADIUS,
)


class TestFleeingTargetObstacleAvoidance:
    """Tests for obstacle avoidance in FleeingTargetStrategy."""
    
    def test_target_moves_away_from_obstacle_if_generated_too_close(self):
        """
        Test 1: Уход от препятствия если генерация в опасной близости.
        
        Если цель сгенерирована слишком близко к препятствию, она должна
        автоматически переместиться в валидную позицию.
        """
        # Создаем стратегию
        strategy = FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=np.random.default_rng(42)
        )
        
        # Создаем препятствие
        obstacle = np.array([100.0, 100.0], dtype=np.float64)
        
        # Цель слишком близко к препятствию (ближе чем TARGET_MIN_DISTANCE_TO_OBSTACLE)
        too_close_distance = TARGET_MIN_DISTANCE_TO_OBSTACLE * 0.5
        direction = np.array([1.0, 0.0])  # Направление от препятствия
        direction = direction / np.linalg.norm(direction)
        target_pos = obstacle + direction * too_close_distance
        
        # Проверяем, что начальная позиция слишком близко
        initial_distance = np.linalg.norm(target_pos - obstacle)
        assert initial_distance < TARGET_MIN_DISTANCE_TO_OBSTACLE, \
            "Test setup: target should be too close to obstacle"
        
        # Валидируем позицию
        valid_position = strategy.ensure_valid_position(
            target_pos.copy(),
            [obstacle],
            TARGET_MIN_DISTANCE_TO_OBSTACLE
        )
        
        # Проверяем, что позиция была скорректирована
        final_distance = np.linalg.norm(valid_position - obstacle)
        assert final_distance >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
            f"Target should be at least {TARGET_MIN_DISTANCE_TO_OBSTACLE}m from obstacle, " \
            f"got {final_distance}m"
        
        # Проверяем, что позиция в пределах поля
        assert np.all(valid_position >= 0.0) and np.all(valid_position <= FIELD_SIZE), \
            "Target position should be within field boundaries"
    
    def test_target_does_not_move_into_obstacle_danger_zone(self):
        """
        Test 2: Перемещение не должно идти в опасную зону препятствий.
        
        При движении от дрона цель не должна попадать в опасную зону препятствий.
        """
        strategy = FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=np.random.default_rng(42)
        )
        
        # Создаем сценарий: дрон слева, препятствие справа, цель между ними
        drone_pos = np.array([50.0, 100.0], dtype=np.float64)
        obstacle_pos = np.array([150.0, 100.0], dtype=np.float64)
        target_pos = np.array([100.0, 100.0], dtype=np.float64)  # Между дроном и препятствием
        
        # Проверяем, что цель достаточно далеко от препятствия изначально
        initial_distance_to_obstacle = np.linalg.norm(target_pos - obstacle_pos)
        assert initial_distance_to_obstacle >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
            "Test setup: target should start at safe distance from obstacle"
        
        env_data = EnvData(
            drone=Drone(pos=drone_pos, orientation=0.0),
            target_position=target_pos.copy(),
            obstacles=[obstacle_pos]
        )
        
        # Выполняем несколько шагов движения
        current_pos = target_pos.copy()
        for step in range(10):
            new_pos = strategy.update(
                current_pos.copy(),
                env_data,
                DELTA_T
            )
            
            # Проверяем, что новая позиция не слишком близко к препятствию
            distance_to_obstacle = np.linalg.norm(new_pos - obstacle_pos)
            assert distance_to_obstacle >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
                f"Step {step}: Target moved too close to obstacle. " \
                f"Distance: {distance_to_obstacle}m, required: {TARGET_MIN_DISTANCE_TO_OBSTACLE}m"
            
            # Обновляем позицию дрона (он приближается к цели)
            env_data.drone.position += np.array([1.0, 0.0]) * DELTA_T * 5.0
            current_pos = new_pos
    
    def test_target_corrects_movement_when_fleeing_would_hit_obstacle(self):
        """
        Test 3: Коррекция движения если при движении от дрона цель должна двигаться на препятствие.
        
        Если направление убегания от дрона ведет к препятствию, цель должна
        скорректировать траекторию, чтобы избежать столкновения.
        """
        strategy = FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=np.random.default_rng(42)
        )
        
        # Создаем сценарий, где убегание от дрона ведет прямо к препятствию
        # Дрон слева, препятствие справа, цель между ними
        drone_pos = np.array([50.0, 100.0], dtype=np.float64)
        # Place the obstacle so that:
        # - target starts at a safe distance (>= TARGET_MIN_DISTANCE_TO_OBSTACLE)
        # - one naive flee step would enter the danger zone (< TARGET_MIN_DISTANCE_TO_OBSTACLE)
        #
        # With TARGET_MAX_SPEED*DELTA_T = 1.0, target would move from x=100 -> x=101.
        # Choose obstacle_x = 101.7 => start distance = 1.7 (safe), after move distance = 0.7 (unsafe).
        obstacle_pos = np.array([101.7, 100.0], dtype=np.float64)
        target_pos = np.array([100.0, 100.0], dtype=np.float64)
        
        # Направление убегания от дрона (вправо) ведет прямо к препятствию
        flee_direction = np.array([1.0, 0.0])  # Вправо
        expected_collision_pos = target_pos + flee_direction * TARGET_MAX_SPEED * DELTA_T
        
        # Проверяем, что без коррекции цель попала бы в опасную зону
        distance_to_obstacle_if_no_correction = np.linalg.norm(expected_collision_pos - obstacle_pos)
        assert distance_to_obstacle_if_no_correction < TARGET_MIN_DISTANCE_TO_OBSTACLE, \
            "Test setup: fleeing direction should lead to obstacle danger zone"
        
        env_data = EnvData(
            drone=Drone(pos=drone_pos, orientation=0.0),
            target_position=target_pos.copy(),
            obstacles=[obstacle_pos]
        )
        
        # Выполняем шаг движения
        new_pos = strategy.update(
            target_pos.copy(),
            env_data,
            DELTA_T
        )

        # Движение должно быть "аккуратным": не превышать скорость * время (без телепорта)
        move_dist = np.linalg.norm(new_pos - target_pos)
        assert move_dist <= TARGET_MAX_SPEED * DELTA_T * 1.1, \
            f"Movement too large (teleport?). dist={move_dist}, max={TARGET_MAX_SPEED * DELTA_T}"
        
        # Проверяем, что позиция была скорректирована
        final_distance_to_obstacle = np.linalg.norm(new_pos - obstacle_pos)
        assert final_distance_to_obstacle >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
            f"Target should avoid obstacle. Distance: {final_distance_to_obstacle}m, " \
            f"required: {TARGET_MIN_DISTANCE_TO_OBSTACLE}m"
        
        # Проверяем, что цель все еще движется от дрона (хотя бы частично)
        # или скорректировала направление
        direction_to_drone_before = drone_pos - target_pos
        direction_to_drone_after = drone_pos - new_pos
        
        # Цель должна либо удаляться от дрона, либо скорректировать направление
        # Проверяем, что новая позиция не ближе к дрону, чем могла бы быть без препятствия
        # (или что направление было скорректировано)
        movement = new_pos - target_pos
        movement_norm = np.linalg.norm(movement)
        
        # Движение должно произойти (цель не должна остаться на месте)
        assert movement_norm > 0.01, "Target should move"
        
        # Проверяем, что позиция валидна
        assert np.all(new_pos >= 0.0) and np.all(new_pos <= FIELD_SIZE), \
            "Target position should be within field boundaries"
    
    def test_target_handles_multiple_obstacles(self):
        """Дополнительный тест: цель должна избегать нескольких препятствий."""
        strategy = FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=np.random.default_rng(42)
        )
        
        # Создаем несколько препятствий вокруг цели
        drone_pos = np.array([50.0, 100.0], dtype=np.float64)
        target_pos = np.array([100.0, 100.0], dtype=np.float64)
        
        obstacles = [
            np.array([120.0, 100.0], dtype=np.float64),  # Справа
            np.array([100.0, 120.0], dtype=np.float64),  # Сверху
            np.array([80.0, 100.0], dtype=np.float64),   # Слева
        ]
        
        env_data = EnvData(
            drone=Drone(pos=drone_pos, orientation=0.0),
            target_position=target_pos.copy(),
            obstacles=obstacles
        )
        
        # Выполняем несколько шагов
        current_pos = target_pos.copy()
        for step in range(20):
            new_pos = strategy.update(
                current_pos.copy(),
                env_data,
                DELTA_T
            )
            
            # Проверяем расстояние до каждого препятствия
            for i, obstacle in enumerate(obstacles):
                distance = np.linalg.norm(new_pos - obstacle)
                assert distance >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
                    f"Step {step}, obstacle {i}: Target too close. " \
                    f"Distance: {distance}m, required: {TARGET_MIN_DISTANCE_TO_OBSTACLE}m"
            
            # Обновляем позицию дрона
            env_data.drone.position += np.array([1.0, 0.0]) * DELTA_T * 5.0
            current_pos = new_pos
    
    def test_target_stays_valid_when_on_obstacle_boundary(self):
        """Тест: цель на границе опасной зоны должна оставаться валидной."""
        strategy = FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=np.random.default_rng(42)
        )
        
        obstacle_pos = np.array([100.0, 100.0], dtype=np.float64)
        
        # Цель точно на границе безопасной зоны
        direction = np.array([1.0, 0.0])
        target_pos = obstacle_pos + direction * TARGET_MIN_DISTANCE_TO_OBSTACLE
        
        # Валидируем позицию
        valid_pos = strategy.ensure_valid_position(
            target_pos.copy(),
            [obstacle_pos],
            TARGET_MIN_DISTANCE_TO_OBSTACLE
        )
        
        # Позиция должна остаться валидной (на границе или дальше)
        distance = np.linalg.norm(valid_pos - obstacle_pos)
        assert distance >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
            f"Target on boundary should remain valid. Distance: {distance}m"
    
    def test_target_correction_creates_smooth_movement(self):
        """Тест: коррекция движения должна создавать плавное движение."""
        strategy = FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=np.random.default_rng(42)
        )
        
        # Сценарий: дрон слева, препятствие справа, цель между ними
        drone_pos = np.array([50.0, 100.0], dtype=np.float64)
        obstacle_pos = np.array([110.0, 100.0], dtype=np.float64)
        target_pos = np.array([100.0, 100.0], dtype=np.float64)
        
        env_data = EnvData(
            drone=Drone(pos=drone_pos, orientation=0.0),
            target_position=target_pos.copy(),
            obstacles=[obstacle_pos]
        )
        
        # Выполняем несколько шагов и проверяем плавность движения
        positions = [target_pos.copy()]
        current_pos = target_pos.copy()
        
        for step in range(5):
            new_pos = strategy.update(
                current_pos.copy(),
                env_data,
                DELTA_T
            )
            positions.append(new_pos.copy())
            
            # Проверяем, что движение не слишком резкое
            movement = new_pos - current_pos
            movement_distance = np.linalg.norm(movement)
            
            # Максимальное расстояние за один шаг не должно превышать скорость * время
            max_possible_distance = TARGET_MAX_SPEED * DELTA_T
            assert movement_distance <= max_possible_distance * 1.1, \
                f"Step {step}: Movement too large. " \
                f"Distance: {movement_distance}m, max: {max_possible_distance}m"
            
            # Обновляем позицию дрона
            env_data.drone.position += np.array([1.0, 0.0]) * DELTA_T * 5.0
            current_pos = new_pos
        
        # Проверяем, что все позиции валидны
        for i, pos in enumerate(positions):
            distance_to_obstacle = np.linalg.norm(pos - obstacle_pos)
            assert distance_to_obstacle >= TARGET_MIN_DISTANCE_TO_OBSTACLE, \
                f"Position {i} too close to obstacle. Distance: {distance_to_obstacle}m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
