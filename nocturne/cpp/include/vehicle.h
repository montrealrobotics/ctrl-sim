// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "geometry/vector_2d.h"
#include "object.h"

#include <iostream>

#include "physics/BaseCar.h"

namespace nocturne {

class Vehicle : public Object {
 public:
  Vehicle() = default;

  Vehicle(int64_t id, float length, float width,
          const geometry::Vector2D& position, float heading, float speed,
          const geometry::Vector2D& target_position, float target_heading,
          float target_speed, bool can_block_sight = true,
          bool can_be_collided = true, bool check_collision = true);

  Vehicle(int64_t id, float length, float width, float max_speed,
          const geometry::Vector2D& position, float heading, float speed,
          const geometry::Vector2D& target_position, float target_heading,
          float target_speed, bool can_block_sight = true,
          bool can_be_collided = true, bool check_collision = true)
      : Object(id, length, width, max_speed, position, heading, speed,
               target_position, target_heading, target_speed, can_block_sight,
               can_be_collided, check_collision) { std::cout << "Vehicle (maxspeed)" << std::endl; }

  virtual void Step(float dt);

  // we need the 3 functions for TrajectoryCar (aka. expert_mode in non-physics nocturne):
  // set_position
  // set_heading
  // set_speed
  virtual void set_position(float x, float y);
  virtual void set_position(const geometry::Vector2D& position);
  virtual void set_heading(float heading);
  virtual void set_speed(float speed);

  // functions needed for FreeCar:
  virtual void set_acceleration(float acceleration);
  virtual void set_steering(float steering);
  void brake(float brake);

  virtual void ApplyAction(const Action& action);

  ObjectType Type() const override { return ObjectType::kVehicle; }

  virtual void CreatePhysicsBody(); // (re)create the right physics body type (or nothing if not a vehicle...)


protected:
  physics::BaseCar*  m_PhysicsCar;

};

}  // namespace nocturne
