// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>
#include <cstdint>
#include <string>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {

enum class CollisionType {
  kNotCollided = 0,
  kVehicleVehicleCollision = 1,
  kVehicleRoadEdgeCollision = 2,
};

class ObjectBase : public sf::Drawable, public geometry::AABBInterface {
 public:
  ObjectBase() = default;

  explicit ObjectBase(const geometry::Vector2D& position)
      : position_(position) {}

  ObjectBase(const geometry::Vector2D& position, bool can_block_sight,
             bool can_be_collided, bool check_collision)
      : position_(position),
        can_block_sight_(can_block_sight),
        can_be_collided_(can_be_collided),
        check_collision_(check_collision) {}

  const geometry::Vector2D& position() const { return position_; }
  virtual void set_position(const geometry::Vector2D& position) {
    position_ = position;
  }
  virtual void set_position(float x, float y) { position_ = geometry::Vector2D(x, y); }

  bool can_block_sight() const { return can_block_sight_; }
  bool can_be_collided() const { return can_be_collided_; }
  bool check_collision() const { return check_collision_; }

  bool collided() const { return collided_; }
  void set_collided(bool collided) { collided_ = collided; }

  CollisionType collision_type() const { return collision_type_; }
  CollisionType collision_type_edge() const { return collision_type_edge_; }
  CollisionType collision_type_veh() const { return collision_type_veh_; }

  void set_collision_type(CollisionType collision_type) {
    if (collision_type == CollisionType::kVehicleVehicleCollision){
      collision_type_veh_ = collision_type;
    }
    else if (collision_type == CollisionType::kVehicleRoadEdgeCollision){
      collision_type_edge_ = collision_type;
    }

    collision_type_ = collision_type;
  }

  void ResetCollision() {
    collided_ = false;
    collision_type_ = CollisionType::kNotCollided;
    collision_type_veh_ = CollisionType::kNotCollided;
    collision_type_edge_ = CollisionType::kNotCollided;
  }

  virtual float Radius() const = 0;

  virtual geometry::ConvexPolygon BoundingPolygon() const = 0;

  geometry::AABB GetAABB() const override {
    return BoundingPolygon().GetAABB();
  }

 protected:
  geometry::Vector2D position_;

  const bool can_block_sight_ = false;
  const bool can_be_collided_ = false;
  const bool check_collision_ = false;
  bool collided_ = false;
  CollisionType collision_type_ = CollisionType::kNotCollided;
  CollisionType collision_type_veh_ = CollisionType::kNotCollided;
  CollisionType collision_type_edge_ = CollisionType::kNotCollided;
};

}  // namespace nocturne
