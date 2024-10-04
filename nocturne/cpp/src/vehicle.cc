#include "vehicle.h"

#include <iostream>
#include <math.h>

#include "physics/FreeCar.h"
#include "physics/ExpertControlCar.h"
#include "physics/Singletons.h"

namespace nocturne {

    Vehicle::Vehicle(int64_t id, float length, float width,
          const geometry::Vector2D& position, float heading, float speed,
          const geometry::Vector2D& target_position, float target_heading,
          float target_speed, bool can_block_sight,
          bool can_be_collided, bool check_collision)
      : Object(id, length, width, position, heading, speed, target_position,
               target_heading, target_speed, can_block_sight, can_be_collided,
               check_collision) 
    { 
//        std::cout << "Vehicle id=" << id << " can_be_collided=" << can_be_collided << " check_collision=" << check_collision << std::endl; 
        m_PhysicsCar = NULL;
    }

    void Vehicle::Step(float dt)
    {
        // TODO: physics update
        if (physics_simulated_)
        {
            // first scenario:
            // Physics update
            
            // Note: Physics Step is not done here, but from m_PhysicsCar

            if (expert_control_)
            {
                // vehicle follows a predefined route 

                // m_PhysicsCar is a ExpertControlCar
                // note: looks like this Step() function is never called in this case
                b2Vec2 pos = m_PhysicsCar->GetPosition(); // debug only
//                std::cout << "ExpertControlCar id=" << id_ << " Step dt=" << dt << " pos=" << pos.x << ":" << pos.y << std::endl; 
                Object::Step(dt);
            }
            else
            {
                // vehicle is a "free car"
                // m_PhysicsCar is a FreeCar

                b2Vec2 pos = m_PhysicsCar->GetPosition();
                // std::cout << "FreeCar id=" << id_ << " Step dt=" << dt << " pos=" << pos.x << ":" << pos.y << std::endl; 
                Object::set_position(geometry::Vector2D(pos.x,pos.y));
                Object::set_speed(m_PhysicsCar->GetSpeed());
                Object::set_heading(m_PhysicsCar->GetAngle()+M_PI*0.5f);
            }


        }
        else
        {
            // second scenario:
            // "old-fashined" update
            Object::Step(dt);   // temp
        }

    }


    void Vehicle::ApplyAction(const Action& action)
    {
//        std::cout << "Vehicle id=" << id_ << " ApplyAction" << std::endl; 
        Object::ApplyAction(action);
    }

    void Vehicle::set_position(float x, float y)
    {
//        std::cout << "Vehicle id=" << id_ << " set_position(x,y) " << x << ":" << y << std::endl; 
        Object::set_position(x,y);
        if (m_PhysicsCar) m_PhysicsCar->SetPosition(b2Vec2(x,y));
    }

    void Vehicle::set_position(const geometry::Vector2D& position)
    {
//        std::cout << "Vehicle id=" << id_ << " set_position(Vector2D) " << position.x() << ":" << position.y() << std::endl; 
        Object::set_position(position);
        if (m_PhysicsCar) m_PhysicsCar->SetPosition(b2Vec2(position.x(),position.y()));
    }

    void Vehicle::set_heading(float heading)
    {
//        std::cout << "Vehicle id=" << id_ << " set_heading " << heading << std::endl; 
        Object::set_heading(heading);
        if (m_PhysicsCar) m_PhysicsCar->SetAngle(heading_-M_PI*0.5f);
    }

    void Vehicle::set_speed(float speed)
    {
//        std::cout << "Vehicle id=" << id_ << " set_speed " << speed << std::endl; 
        Object::set_speed(speed);
        if (m_PhysicsCar)
        {
            float c = cosf(heading_);    
            float s = sinf(heading_);    
            b2Vec2 speed_v(speed*c,speed*s);
            m_PhysicsCar->SetSpeed(speed_v);
        }
    }

    void Vehicle::set_acceleration(float acceleration)
    {
//        std::cout << "Vehicle id=" << id_ << " set_acceleration " << acceleration << std::endl; 
        physics::FreeCar* freecar = dynamic_cast<physics::FreeCar*>(m_PhysicsCar);
        if (freecar)
            freecar->Throttle(acceleration);
        else
            Object::set_acceleration(acceleration);
    }

    void Vehicle::set_steering(float steering)
    {
//        std::cout << "Vehicle id=" << id_ << " set_steering " << steering << std::endl;

        Object::set_steering(steering);
        physics::FreeCar* freecar = dynamic_cast<physics::FreeCar*>(m_PhysicsCar);
        if (freecar)
            freecar->Turn(steering);
    }

    void Vehicle::brake(float brake)
    {
//        std::cout << "Vehicle id=" << id_ << " set_acceleration " << acceleration << std::endl; 
        physics::FreeCar* freecar = dynamic_cast<physics::FreeCar*>(m_PhysicsCar);
        if (freecar)
            freecar->Brake(brake);
    }

    void Vehicle::CreatePhysicsBody()
    {
        // remove previous physics object
        if (m_PhysicsCar) 
        {
            physics::GetPhysicsSimulation()->RemoveCar(m_PhysicsCar);
            delete (m_PhysicsCar);
        }
        m_PhysicsCar = nullptr;

//        std::cout << "Vehicle id=" << id_ << " CreatePhysicsBody physics_simulated_=" << physics_simulated_ << " expert_control=" << expert_control_ << " W=" << width_ << " L=" << length_ << std::endl; 

        if (physics_simulated_)
        {
            // TODO:
            // physics car size
            // car parameters (mass, accelerations, ...)
            
            // position: Geometry::Vector2D position_
            // heading : float heading_
            // size : float length_, float width_

            // which object to create ?
            if (expert_control_)
            {
                // create an expert controlled car
                m_PhysicsCar = new physics::ExpertControlCar(width_,length_);
            }
            else
            {
                // create a free car
                m_PhysicsCar = new physics::FreeCar(width_,length_);
            }
            m_PhysicsCar->SetAngle(heading_-M_PI*0.5f);
            m_PhysicsCar->SetPosition(b2Vec2(position_.x(),position_.y()));
            float c = cosf(heading_);    
            float s = sinf(heading_);    
            b2Vec2 speed_v(speed_*c,speed_*s);
            m_PhysicsCar->SetSpeed(speed_v);
            
            physics::GetPhysicsSimulation()->AddCar(m_PhysicsCar);
        }
    }

}



