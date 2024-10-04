#pragma once

#include <box2d/box2d.h>
#include "PhysicsSimulation.h"

namespace physics {

// singletons
b2World *Getb2World();  
PhysicsSimulation *GetPhysicsSimulation();  


}   // namespace physics
