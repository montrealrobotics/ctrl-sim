#include "Singletons.h"

namespace physics {

static b2World *s_b2World = 0;
b2World* Getb2World()
{
    if (!s_b2World) 
    {
        b2Vec2 gravity;
        gravity.Set(0.0f, 0.0f);
        s_b2World = new b2World(gravity);
    }
    return s_b2World;
}

static PhysicsSimulation *s_PhysicsSimulation = 0;
PhysicsSimulation* GetPhysicsSimulation()
{
    if (!s_PhysicsSimulation) 
    {
        s_PhysicsSimulation = new PhysicsSimulation();
    }
    return s_PhysicsSimulation;
}

}   // namespace physics
