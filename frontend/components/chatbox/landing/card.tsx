type CardProps = {
  boldContent: string;
  normalContent: string;
};

const Card = (props: CardProps) => {
  return (
    <div className="flex flex-col items-start justify-center rounded-xl border border-solid border-stone-300 px-5 py-4 text-center">
      <div className="text-sm font-semibold leading-4 w-full">{props.boldContent}</div>
      <div className="mt-1.5 text-xs leading-4 w-full">{props.normalContent}</div>
    </div>
  );
};

export default Card;
