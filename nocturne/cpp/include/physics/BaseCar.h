#pragma once

#include <box2d/box2d.h>

namespace physics {

class BaseCar
{
    public:
        BaseCar(float width,float length);
        virtual ~BaseCar();

        virtual void Step(float dt) = 0;

        b2Vec2 GetPosition();
        float GetAngle();
        float GetSpeed();

        void SetPosition(b2Vec2 pos);
        void SetAngle(float angle);
        void SetSpeed(b2Vec2 speed);

    protected:
        b2Body *m_Body;
        float m_Width;
        float m_Length;
};

}   // namespace physics
